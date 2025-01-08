import logging
import sys

class RefreshingConsoleHandler(logging.StreamHandler):
    """
    A custom logging handler that refreshes multiple lines in the console for certain log messages.
    Each log message intended for refreshing should contain '\n' characters to indicate multiple lines.
    """
    def __init__(self, stream=None):
        super().__init__(stream)
        self.previous_line_count = 0

    def emit(self, record):
        try:
            msg = self.format(record)
            if getattr(record, 'refresh', False):
                # Split message into lines
                lines = msg.split('\n')
                line_count = len(lines)
                
                # Move cursor up by the previous number of lines
                if self.previous_line_count > 0:
                    sys.stdout.write(f'\033[{self.previous_line_count}A')
                
                # Clear each of the previous lines
                for _ in range(self.previous_line_count):
                    sys.stdout.write('\033[K\n')  # Clear line and move to next line
                if self.previous_line_count > 0:
                    # Move cursor up again to overwrite
                    sys.stdout.write(f'\033[{self.previous_line_count}A')
                
                # Write the new lines
                for line in lines:
                    sys.stdout.write(line + '\n')
                sys.stdout.flush()
                
                # Update the previous_line_count
                self.previous_line_count = line_count
            else:
                # Normal log message, append as new line
                sys.stdout.write(msg + '\n')
                sys.stdout.flush()
                self.previous_line_count = 0
        except Exception:
            self.handleError(record)
