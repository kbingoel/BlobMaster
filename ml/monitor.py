#!/usr/bin/env python3
"""
External training monitor with Rich TUI.

Platform Requirements:
    - Unix/Linux terminal (uses tty/termios/select for keyboard input)
    - Not compatible with Windows (Windows users should use WSL)

Usage:
    python ml/monitor.py [--status-file PATH] [--refresh-interval SECONDS]

Keyboard controls:
    p - Pause training (writes control signal)
    q - Quit monitor (training continues)
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any
import sys
import logging

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn

logger = logging.getLogger(__name__)


class TrainingMonitor:
    """
    External monitor for training progress.

    Platform: Unix/Linux only (requires tty/termios/select)
    """

    def __init__(
        self,
        status_file: str = "models/checkpoints/status.json",
        control_file: str = "models/checkpoints/control.signal",
        refresh_interval: float = 5.0
    ):
        """
        Args:
            status_file: Path to status JSON file
            control_file: Path to control signal file
            refresh_interval: Seconds between status updates
        """
        self.status_file = Path(status_file)
        self.control_file = Path(control_file)
        self.refresh_interval = refresh_interval
        self.console = Console()
        self.last_status: Optional[Dict[str, Any]] = None

    def read_status(self) -> Optional[Dict[str, Any]]:
        """
        Read current status from JSON file.

        Returns:
            Status dict or None if file doesn't exist or is invalid
        """
        if not self.status_file.exists():
            return None

        try:
            with open(self.status_file) as f:
                status = json.load(f)
                return status
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read status: {e}")
            # Return cached status to avoid flickering display
            return self.last_status

    def send_pause_signal(self):
        """Write PAUSE signal to control file."""
        try:
            self.control_file.parent.mkdir(parents=True, exist_ok=True)
            self.control_file.write_text("PAUSE\n")
            self.console.print(
                "[green]✓ Pause signal sent. Training will stop after current iteration.[/green]"
            )
        except OSError as e:
            self.console.print(f"[red]✗ Failed to write pause signal: {e}[/red]")

    def build_display(self, status: Optional[Dict[str, Any]]) -> Layout:
        """
        Build Rich layout from status dict.

        Args:
            status: Status dict or None if not available

        Returns:
            Rich Layout object
        """
        layout = Layout()

        if status is None:
            layout.update(
                Panel(
                    "[yellow]Waiting for training to start...\n"
                    f"Looking for: {self.status_file}\n\n"
                    "[dim]Start training in another terminal:[/dim]\n"
                    "[cyan]python ml/train.py --iterations 500[/cyan]",
                    title="BlobMaster Training Monitor",
                    border_style="yellow"
                )
            )
            return layout

        # Extract data from status (all validated with fallbacks)
        iteration = status.get('iteration', 0)  # 1-indexed from status file
        total = status.get('total_iterations', 500)
        progress_pct = status.get('progress', 0.0)
        eta_days = status.get('eta_days', 0.0)
        eta_hours = status.get('eta_hours', 0.0)
        phase = status.get('phase', 'unknown')
        mcts = status.get('mcts_config', 'N/A')
        elo = status.get('elo')
        elo_change = status.get('elo_change')
        lr = status.get('learning_rate')
        loss = status.get('loss')
        policy_loss = status.get('policy_loss')
        value_loss = status.get('value_loss')
        units = status.get('training_units_generated')
        unit_type = status.get('unit_type', 'rounds')
        tau = status.get('pi_target_tau')
        elapsed_hours = status.get('elapsed_hours', 0.0)

        # Build metrics table
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")

        # Progress (iteration already 1-indexed from status file)
        table.add_row("Iteration", f"{iteration}/{total} ({progress_pct:.1%})")

        # Phase & MCTS
        table.add_row("Phase", f"{phase} ({mcts} MCTS)")

        # ETA
        if eta_days >= 1.0:
            eta_str = f"{eta_days:.1f} days ({eta_hours:.1f} hours)"
        else:
            eta_str = f"{eta_hours:.1f} hours"
        table.add_row("ETA", eta_str)

        # Elapsed
        if elapsed_hours >= 24:
            elapsed_str = f"{elapsed_hours/24:.1f} days ({elapsed_hours:.1f} hours)"
        else:
            elapsed_str = f"{elapsed_hours:.1f} hours"
        table.add_row("Elapsed", elapsed_str)

        # ELO (if available)
        if elo is not None:
            elo_str = f"{elo:.0f}"
            if elo_change is not None:
                sign = "+" if elo_change >= 0 else ""
                elo_str += f" ({sign}{elo_change:.0f})"
            table.add_row("ELO", elo_str)

        # Learning rate
        if lr is not None:
            table.add_row("Learning Rate", f"{lr:.6f}")

        # Losses
        if loss is not None:
            loss_str = f"{loss:.4f}"
            if policy_loss is not None and value_loss is not None:
                loss_str += f" (π:{policy_loss:.3f} v:{value_loss:.3f})"
            table.add_row("Loss", loss_str)

        # Training units
        if units is not None:
            table.add_row(f"{unit_type.capitalize()}/iter", f"{units:,}")

        # Target temperature
        if tau is not None:
            table.add_row("Target τ", f"{tau:.3f}")

        # Progress bar
        progress_bar = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )
        progress_task = progress_bar.add_task(
            "Overall Progress",
            total=100,
            completed=progress_pct * 100
        )

        # Build layout
        layout.split_column(
            Layout(
                Panel(table, title="Training Status", border_style="green"),
                size=22
            ),
            Layout(
                Panel(progress_bar, title="Progress", border_style="blue"),
                size=5
            ),
            Layout(
                Panel(
                    "[cyan][p][/cyan] Pause training  |  [cyan][q][/cyan] Quit monitor  |  "
                    f"[dim]Refresh: {self.refresh_interval}s[/dim]",
                    border_style="yellow"
                ),
                size=3
            )
        )

        return layout

    def run(self):
        """
        Run monitor loop with keyboard input.

        Platform: Unix/Linux only (uses select/tty/termios)
        """
        # Check platform compatibility
        try:
            import select
            import tty
            import termios
        except ImportError:
            self.console.print(
                "[red]Error: Monitor requires Unix/Linux terminal (uses tty/termios/select)[/red]\n"
                "[yellow]Windows users: Please use WSL (Windows Subsystem for Linux)[/yellow]"
            )
            return 1

        # Set terminal to non-blocking mode for keyboard input
        old_settings = termios.tcgetattr(sys.stdin.fileno())
        try:
            tty.setcbreak(sys.stdin.fileno())

            with Live(self.build_display(None), refresh_per_second=1) as live:
                while True:
                    # Read status
                    status = self.read_status()
                    if status is not None:
                        self.last_status = status

                    # Update display
                    live.update(self.build_display(status))

                    # Check for keyboard input (non-blocking with timeout)
                    start_time = time.time()
                    while time.time() - start_time < self.refresh_interval:
                        # Check if input is available (100ms timeout)
                        if select.select([sys.stdin], [], [], 0.1)[0]:
                            key = sys.stdin.read(1).lower()

                            if key == 'q':
                                self.console.print(
                                    "\n[yellow]Exiting monitor. Training continues in background.[/yellow]"
                                )
                                return 0

                            elif key == 'p':
                                self.send_pause_signal()
                                time.sleep(1.5)  # Give user time to see confirmation

                        # Small sleep to avoid busy-waiting
                        time.sleep(0.05)

        except KeyboardInterrupt:
            self.console.print(
                "\n[yellow]Monitor interrupted (Ctrl+C). Training continues in background.[/yellow]"
            )
            return 0

        finally:
            # Always restore terminal settings
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)

        return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="External training monitor with Rich TUI (Unix/Linux only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Platform Requirements:
  Unix/Linux terminal only (uses tty/termios/select)
  Windows users: Use WSL (Windows Subsystem for Linux)

Keyboard controls:
  p - Pause training (writes control signal)
  q - Quit monitor (training continues)

Example usage:
  python ml/monitor.py
  python ml/monitor.py --refresh-interval 10
  python ml/monitor.py --status-file custom/status.json
        """
    )

    parser.add_argument(
        '--status-file',
        type=str,
        default='models/checkpoints/status.json',
        help='Path to status JSON file (default: models/checkpoints/status.json)'
    )

    parser.add_argument(
        '--control-file',
        type=str,
        default='models/checkpoints/control.signal',
        help='Path to control signal file (default: models/checkpoints/control.signal)'
    )

    parser.add_argument(
        '--refresh-interval',
        type=float,
        default=5.0,
        help='Seconds between status updates (default: 5.0)'
    )

    args = parser.parse_args()

    # Create and run monitor
    monitor = TrainingMonitor(
        status_file=args.status_file,
        control_file=args.control_file,
        refresh_interval=args.refresh_interval
    )

    try:
        exit_code = monitor.run()
        return exit_code
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
