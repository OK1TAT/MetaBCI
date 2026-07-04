from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


CYTON_GAIN_X24_SCALE_FACTOR = 0.022351744455307063
CYTON_GAIN_X6_SCALE_FACTOR = CYTON_GAIN_X24_SCALE_FACTOR * 4.0
DEFAULT_SCALE_FACTOR = CYTON_GAIN_X6_SCALE_FACTOR
GAIN_SCALE_FACTORS = {
    f"x{gain}": CYTON_GAIN_X24_SCALE_FACTOR * (24.0 / gain)
    for gain in (1, 2, 4, 6, 8, 12, 24)
}


@dataclass
class AppConfig:
    # Network
    host: str = "192.168.4.1"
    port: int = 9000
    protocol: str = "wifi_shield"  # tcp / udp / wifi_shield

    # Device data format
    channels: int = 16  # 8 or 16
    device_channels: Optional[int] = None  # auto-detected from shield /board
    force_input_channels: int = 16  # 0=auto, else 8 or 16 for decoder input pairing
    sample_rate: int = 500
    endian: str = "big"  # auto / big / little
    scale_factor: float = DEFAULT_SCALE_FACTOR

    # Display and save
    display_seconds: float = 8.0
    ui_fps: int = 25
    save_dir: str = "records"
    subject: str = "subject01"
    save_npz: bool = True
    save_on_start: bool = True
    marker_sync_delay_s: float = 0.25

    # Thread queues
    raw_queue_size: int = 5000
    plot_queue_size: int = 4000
    save_queue_size: int = 12000
    save_batch_size: int = 200
    max_plot_drain_per_tick: int = 1500

    # Socket and reconnect
    socket_timeout: float = 1.0
    reconnect_interval: float = 2.0
    udp_disconnect_timeout: float = 3.0
    wifi_http_port: int = 80
    wifi_local_ip: str = ""
    wifi_output: str = "raw"  # raw or json
    wifi_delimiter: bool = False
    wifi_latency_us: int = 10000
    wifi_accept_timeout: float = 20.0
    wifi_apply_channel_settings: bool = True
    wifi_channel_gain: str = "x6"
    wifi_gain_command: str = "1"

    def validate(self) -> None:
        self.protocol = self.protocol.lower()
        self.endian = self.endian.lower()
        self.wifi_output = self.wifi_output.lower()
        self.wifi_channel_gain = self.wifi_channel_gain.lower()
        if self.protocol not in {"tcp", "udp", "wifi_shield"}:
            raise ValueError("protocol must be tcp, udp or wifi_shield")
        if self.channels not in {8, 16}:
            raise ValueError("channels must be 8 or 16")
        if self.device_channels is not None and self.device_channels not in {8, 16}:
            raise ValueError("device_channels must be 8 or 16 when provided")
        if self.force_input_channels not in {0, 8, 16}:
            raise ValueError("force_input_channels must be 0, 8 or 16")
        if self.endian not in {"auto", "big", "little"}:
            raise ValueError("endian must be auto, big or little")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        if self.port <= 0 or self.port > 65535:
            raise ValueError("port must be 1..65535")
        if self.wifi_http_port <= 0 or self.wifi_http_port > 65535:
            raise ValueError("wifi_http_port must be 1..65535")
        if self.wifi_output not in {"raw", "json"}:
            raise ValueError("wifi_output must be raw or json")
        if self.wifi_channel_gain not in {"x1", "x2", "x4", "x6", "x8", "x12", "x24"}:
            raise ValueError("wifi_channel_gain must be x1, x2, x4, x6, x8, x12, or x24")
        if self.wifi_latency_us < 50:
            raise ValueError("wifi_latency_us must be >= 50")
        if self.wifi_accept_timeout <= 0:
            raise ValueError("wifi_accept_timeout must be > 0")
        if self.display_seconds <= 0:
            raise ValueError("display_seconds must be > 0")
        if self.ui_fps <= 0:
            raise ValueError("ui_fps must be > 0")
        if self.raw_queue_size <= 0 or self.plot_queue_size <= 0 or self.save_queue_size <= 0:
            raise ValueError("queue sizes must be > 0")
        if self.scale_factor <= 0:
            raise ValueError("scale_factor must be > 0")
        if self.save_batch_size <= 0:
            raise ValueError("save_batch_size must be > 0")
        if self.marker_sync_delay_s < 0:
            raise ValueError("marker_sync_delay_s must be >= 0")
        self.save_dir = str(Path(self.save_dir))

    @property
    def buffer_samples(self) -> int:
        return max(1, int(self.sample_rate * self.display_seconds))

    @classmethod
    def from_cli(cls) -> "AppConfig":
        parser = argparse.ArgumentParser(
            description="WiFi EEG receiver/decoder/viewer (8ch/16ch)"
        )
        parser.add_argument(
            "--host",
            type=str,
            default="192.168.4.1",
            help="For wifi_shield/tcp: device IP; for udp: bind host",
        )
        parser.add_argument("--port", type=int, default=9000, help="TCP/UDP port")
        parser.add_argument(
            "--protocol",
            type=str,
            default="wifi_shield",
            choices=["tcp", "udp", "wifi_shield"],
        )
        parser.add_argument("--channels", type=int, default=16, choices=[8, 16])
        parser.add_argument("--sample-rate", type=int, default=500)
        parser.add_argument("--endian", type=str, default="big", choices=["auto", "big", "little"])
        parser.add_argument(
            "--force-input-channels",
            type=int,
            default=16,
            choices=[0, 8, 16],
            help="0=auto detect, 8=force single-packet, 16=force two-packet pairing",
        )
        parser.add_argument(
            "--scale-factor",
            type=float,
            default=None,
            help="uV/count scale factor. Defaults to the selected WiFi channel gain.",
        )
        parser.add_argument("--display-seconds", type=float, default=8.0)
        parser.add_argument("--save-dir", type=str, default="records")
        parser.add_argument("--subject", type=str, default="subject01")
        parser.add_argument("--ui-fps", type=int, default=25)
        parser.add_argument("--save-batch-size", type=int, default=200)
        parser.add_argument("--reconnect-interval", type=float, default=2.0)
        parser.add_argument("--socket-timeout", type=float, default=1.0)
        parser.add_argument("--udp-disconnect-timeout", type=float, default=3.0)
        parser.add_argument("--wifi-http-port", type=int, default=80)
        parser.add_argument("--wifi-local-ip", type=str, default="")
        parser.add_argument("--wifi-output", type=str, default="raw", choices=["raw", "json"])
        parser.add_argument("--wifi-delimiter", action="store_true")
        parser.add_argument("--wifi-latency-us", type=int, default=10000)
        parser.add_argument("--wifi-accept-timeout", type=float, default=20.0)
        parser.add_argument(
            "--wifi-channel-gain",
            type=str,
            default="x6",
            choices=["x1", "x2", "x4", "x6", "x8", "x12", "x24"],
            help="Gain applied to all 16 channels before WiFi streaming.",
        )
        parser.add_argument(
            "--wifi-gain-command",
            type=str,
            default="1",
            help="Device-specific custom command sent before standard channel settings. Use empty string to skip.",
        )
        parser.add_argument(
            "--no-wifi-channel-settings",
            dest="wifi_apply_channel_settings",
            action="store_false",
            help="Do not send OpenBCI channel settings before WiFi streaming.",
        )
        parser.add_argument("--raw-queue-size", type=int, default=5000)
        parser.add_argument("--plot-queue-size", type=int, default=4000)
        parser.add_argument("--save-queue-size", type=int, default=12000)
        parser.add_argument("--max-plot-drain-per-tick", type=int, default=1500)
        parser.add_argument("--save-npz", action="store_true")
        parser.add_argument("--no-save-npz", dest="save_npz", action="store_false")
        parser.set_defaults(save_npz=True, wifi_apply_channel_settings=True)
        args = parser.parse_args()

        config = cls(
            host=args.host,
            port=args.port,
            protocol=args.protocol,
            channels=args.channels,
            sample_rate=args.sample_rate,
            endian=args.endian,
            force_input_channels=args.force_input_channels,
            scale_factor=(
                args.scale_factor
                if args.scale_factor is not None
                else GAIN_SCALE_FACTORS[args.wifi_channel_gain]
            ),
            display_seconds=args.display_seconds,
            ui_fps=args.ui_fps,
            save_dir=args.save_dir,
            subject=args.subject,
            save_npz=args.save_npz,
            save_batch_size=args.save_batch_size,
            reconnect_interval=args.reconnect_interval,
            socket_timeout=args.socket_timeout,
            udp_disconnect_timeout=args.udp_disconnect_timeout,
            wifi_http_port=args.wifi_http_port,
            wifi_local_ip=args.wifi_local_ip,
            wifi_output=args.wifi_output,
            wifi_delimiter=args.wifi_delimiter,
            wifi_latency_us=args.wifi_latency_us,
            wifi_accept_timeout=args.wifi_accept_timeout,
            wifi_apply_channel_settings=args.wifi_apply_channel_settings,
            wifi_channel_gain=args.wifi_channel_gain,
            wifi_gain_command=args.wifi_gain_command,
            raw_queue_size=args.raw_queue_size,
            plot_queue_size=args.plot_queue_size,
            save_queue_size=args.save_queue_size,
            max_plot_drain_per_tick=args.max_plot_drain_per_tick,
        )
        config.validate()
        return config
