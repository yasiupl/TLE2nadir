import argparse
import requests
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime, timedelta, timezone
from skyfield.api import EarthSatellite, load


# Utility Functions
def normalize(vector):
    """Normalize a vector."""
    return vector / np.linalg.norm(vector)


def julian_date_to_mjd(jd):
    """Convert Julian Date to Modified Julian Date."""
    mjd_epoch = 2400000.5  # MJD epoch starting from Jan 1, 1958
    delta_days = jd - mjd_epoch
    days = int(delta_days)
    seconds = (delta_days - days) * 86400
    return days, seconds

def mjd_to_datetime(mjd):
    """
    Convert an MJD (Modified Julian Date) to a naive datetime object (no timezone), discarding fractional seconds.
    """
    mjd_epoch = datetime(1858, 11, 17, tzinfo=timezone.utc)  # MJD epoch
    total_seconds = int(mjd * 86400)  # Convert MJD to seconds and truncate fractions
    utc_datetime = mjd_epoch + timedelta(seconds=total_seconds)
    return utc_datetime.replace(tzinfo=None)  # Remove timezone info


def parse_mjd_pair(value):
    """
    Custom argument parser for a pair of INT and FLOAT (e.g., "60688 38356.397000").
    """
    try:
        # Split the input and convert to INT and FLOAT
        mjd_days, mjd_seconds = value.split()
        return int(mjd_days), float(mjd_seconds)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"'{value}' is not a valid pair of INT and FLOAT. Example: '60688 38356.397000'"
        )

def generate_ccsds_quaternion_data(days, seconds, quaternion):
    """Generate a CCSDS-compatible quaternion string."""
    return f"{days}\t{seconds:.6f}\t{quaternion[0]:.6f}\t{quaternion[1]:.6f}\t{quaternion[2]:.6f}\t{quaternion[3]:.6f}"

# Main logic
def calculate_quaternion(r_sat, v_sat):
    """Calculate quaternion for nadir-pointing satellite."""
    # Normalize position and velocity vectors
    r_unit = normalize(r_sat)
    v_unit = normalize(v_sat)

    # Nadir direction as the opposite of position vector
    nadir_direction = -r_unit

    # X-axis aligns with velocity vector
    x_axis = v_unit

    # Y-axis as the cross product of nadir direction (Z-axis) and X-axis
    y_axis = normalize(np.cross(nadir_direction, x_axis))
    
    # Ensure Z-axis points towards nadir
    z_axis = nadir_direction

    # Rotation matrix from EME2000 to satellite body frame
    R = np.vstack((x_axis, y_axis, z_axis)).T

    # Convert rotation matrix to quaternion
    q = np.zeros(4)
    q[0] = np.sqrt(1.0 + R[0, 0] + R[1, 1] + R[2, 2]) / 2  # Real part
    q[1] = (R[2, 1] - R[1, 2]) / (4 * q[0])
    q[2] = (R[0, 2] - R[2, 0]) / (4 * q[0])
    q[3] = (R[1, 0] - R[0, 1]) / (4 * q[0])

    return q

def generate_quaternions_over_time(satellite, ts, start_time, end_time, interval):
    """Generate quaternion data for a given satellite over time."""
    quaternions = []
    current_time = start_time

    while current_time <= end_time:
        geocentric = satellite.at(ts.utc(current_time.year, current_time.month, current_time.day, 
                                        current_time.hour, current_time.minute, current_time.second))
        r_sat = geocentric.position.km
        v_sat = geocentric.velocity.km_per_s

        jd = ts.utc(current_time.year, current_time.month, current_time.day, 
                    current_time.hour, current_time.minute, current_time.second).tt
        days, seconds = julian_date_to_mjd(jd)

        quaternion = calculate_quaternion(r_sat, v_sat)
        quaternions.append((days, seconds, quaternion))

        current_time += timedelta(seconds=interval)

    return quaternions

# GUI Implementation
class TLE2nadirApp:
    def __init__(self, root, start_date="2025-01-13 00:00:00", end_date="2025-01-20 00:00:00", sampling=60, output_file="output.txt", satellite_id=None):
        self.root = root
        self.root.title("TLE2nadir Generator")

        # Satellite ID Input
        tk.Label(root, text="Satellite NORAD ID:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.norad_id = tk.StringVar(value=satellite_id)
        tk.Entry(root, textvariable=self.norad_id, width=30).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(root, text="Download", command=self.download_tle).grid(row=0, column=2, padx=5, pady=5)

        # TLE Input Fields
        tk.Label(root, text="TLE data:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.tle1 = tk.StringVar()
        self.tle2 = tk.StringVar()
        self.tle1_field = tk.Entry(root, textvariable=self.tle1, width=50)
        self.tle1_field.grid(row=1, column=1, columnspan=2, padx=5, pady=2)
        self.tle1_field.bind("<<Paste>>", self.handle_tle_paste)
        self.tle2_field = tk.Entry(root, textvariable=self.tle2, width=50)
        self.tle2_field.grid(row=2, column=1, columnspan=2, padx=5, pady=2)
 
        # Start/End Date Inputs
        tk.Label(root, text="Start date:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.start_time = tk.StringVar(value=start_date)
        tk.Entry(root, textvariable=self.start_time, width=30).grid(row=3, column=1, padx=5, pady=5)

        tk.Label(root, text="End date:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.end_time = tk.StringVar(value=end_date)
        tk.Entry(root, textvariable=self.end_time, width=30).grid(row=4, column=1, padx=5, pady=5)

        # Sampling Interval
        tk.Label(root, text="Sampling interval:").grid(row=5, column=0, sticky="w", padx=5, pady=5)
        self.interval = tk.StringVar(value=sampling)
        tk.Entry(root, textvariable=self.interval, width=10).grid(row=5, column=1, padx=5, pady=5, sticky="w")
        tk.Label(root, text="seconds").grid(row=5, column=1, padx=5, pady=5)

        # Output File Selection
        tk.Label(root, text="Output file:").grid(row=6, column=0, sticky="w", padx=5, pady=5)
        self.output_file = tk.StringVar(value=output_file)
        tk.Entry(root, textvariable=self.output_file, width=50).grid(row=6, column=1, padx=5, pady=5)
        tk.Button(root, text="Browse", command=self.browse_output_file).grid(row=6, column=2, padx=5, pady=5)

        # Generate Button
        tk.Button(root, text="Generate", command=self.generate_quaternions, bg="green", fg="white").grid(row=7, column=1, pady=10)

    def handle_tle_paste(self, event):
        """
        Handle paste events for TLE1 field. Automatically wraps the TLE text to the second input field.
        """
        try:
            # Get clipboard content
            clipboard_content = self.root.clipboard_get()

            # Split the TLE by newline
            tle_lines = clipboard_content.split("\n")

            # Populate TLE1 and TLE2 fields
            self.tle1_field.delete(0, tk.END)
            self.tle1_field.insert(0, tle_lines[0])  # First line of TLE

            if len(tle_lines) > 1:
                self.tle2_field.delete(0, tk.END)
                self.tle2_field.insert(0, tle_lines[1])  # Second line of TLE
        except tk.TclError:
            #messagebox.showerror("Error", f"Invalid TLE")
            pass  # If clipboard content isn't valid text, do nothing


    def download_tle(self):
        """Download TLE data based on NORAD ID."""
        norad_id = self.norad_id.get()
        if not norad_id.isdigit():
            messagebox.showerror("Error", "Please enter a valid NORAD ID.")
            return
        try:
            tle = requests.get(f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}").text.split('\n')
            self.tle1.set(tle[1])
            self.tle2.set(tle[2])
        except Exception as e:
            messagebox.showerror("Error", f"Could not download TLE: {e}")

    def browse_output_file(self):
        """Select output file path."""
        file_path = filedialog.asksaveasfilename(initialdir=self.browse_output_file, defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
        if file_path:
            self.output_file.set(file_path)

    def generate_quaternions(self):
        """Generate quaternions and save to file."""
        try:
            ts = load.timescale()
            satellite = EarthSatellite(self.tle1.get(), self.tle2.get())
            start_time = datetime.strptime(self.start_time.get(), "%Y-%m-%d %H:%M:%S")
            end_time = datetime.strptime(self.end_time.get(), "%Y-%m-%d %H:%M:%S")
            interval = int(self.interval.get())
            output_path = self.output_file.get()

            quaternions = generate_quaternions_over_time(satellite, ts, start_time, end_time, interval)

            with open(output_path, 'w') as f:

                f.write(f"""CIC_AEM_VERS = 1.0
COMMENT Generated by yasiu.pl TLE2nadir generator
CREATION_DATE = {(datetime.now(timezone.utc)).isoformat(timespec='microseconds')}
ORIGINATOR = VTS

META_START

OBJECT_NAME = {self.norad_id.get() or "UNKNOWN"}
OBJECT_ID = UNKNOWN
REF_FRAME_A = UNDEFINED
REF_FRAME_B = UNDEFINED
ATTITUDE_DIR = A2B
TIME_SYSTEM = UTC
ATTITUDE_TYPE = QUATERNION

META_STOP
\n
""")
                for days, seconds, quaternion in quaternions:
                    f.write(generate_ccsds_quaternion_data(days, seconds, quaternion) + '\n')

            messagebox.showinfo("Success", "Quaternion data saved successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate quaternions: {e}")

def main():

    # Set up the command line parser
    parser = argparse.ArgumentParser(description='Satellite Data Generator for VTS.')

    # Define command line arguments
    parser.add_argument('--vtsgentype', required=True, choices=['GenPosition', 'GenQuaternion', 'GenEuler', 'GenAxis', 'GenAngle', 'GenDirection', 'GenAltAz'],
                        help='The requested generation type')
    parser.add_argument('--vtsdefaultfile', required=True, help='Default output file path')
    parser.add_argument('--vtsmjddates', required=True, nargs=2, type=parse_mjd_pair, metavar=('StartMJD', 'EndMJD'),
                        help='Start and End dates as MJD')

    # Parse the arguments
    args = parser.parse_args()

    if args.vtsgentype != 'GenQuaternion':
        messagebox.showerror("Error", f"Operation {args.vtsgentype} not supported!")
        return

    # Convert MJD pairs to datetime
    start_date = None
    end_date = None
    if args.vtsmjddates:
        start_mjd_days, start_mjd_seconds = args.vtsmjddates[0]
        end_mjd_days, end_mjd_seconds = args.vtsmjddates[1]

        # Combine days and seconds into a full MJD and convert
        start_date = mjd_to_datetime(start_mjd_days + start_mjd_seconds / 86400)
        end_date = mjd_to_datetime(end_mjd_days + end_mjd_seconds / 86400)

    root = tk.Tk()
    app = TLE2nadirApp(
        root,
        start_date=start_date,
        end_date=end_date,
        output_file=args.vtsdefaultfile,
    )
    root.mainloop()

if __name__ == "__main__":
    main()
