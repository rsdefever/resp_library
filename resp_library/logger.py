import datetime
import textwrap


class Logger:
    def __init__(self, log_path, max_width=101):
        self.fh = log_path.open("w")
        self.fh.write(f"RESP performed at: {datetime.utcnow()} UTC\n")
        self.max_width = max_width

    def log(self, string):
        lines = string.splitlines()
        for line in lines:
            if len(line) > self.max_width:
                sub_lines = textwrap.wrap(line, self.max_width)
                for ll in sub_lines:
                    self.fh.write(ll + "\n")
            else:
                self.fh.write(line + "\n")

    def close(self):
        self.fh.write("END\n")
        self.fh.close()
