"""
    VTS Timeloop Generator plugin generating Nadir-pointing quaternion from a satellite TLE over a given time period.
    Copyright (C) 2025  Marcin Jasiukowicz

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

__author__      = "Marcin Jasiukowicz"
__copyright__   = "Copyright (C) 2025  Marcin Jasiukowicz"
__license__     = "GPLv3"
__version__     = "1.0.0"
__email__       = "contact@yasiu.pl"
__status__      = "Production"

import argparse
import requests
import numpy as np
import tkinter as tk
import os # Added for path manipulation
from tkinter import filedialog, messagebox, ttk # Added ttk
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
    """Calculate quaternion for nadir-pointing satellite.
    
    The satellite body frame is defined as:
    - Z-axis: Points towards nadir (opposite of satellite position vector).
    - X-axis: Aligns with the satellite's velocity vector if available and not radial.
              If velocity is zero, not available, or purely radial, a conventional
              X-axis is chosen (typically aligning with an inertial reference direction
              projected onto the plane perpendicular to the Z-axis).
    - Y-axis: Completes the right-handed system (Y = Z x X).
    """
    
    # Epsilon for floating point comparisons (e.g., checking for zero vectors)
    epsilon = 1e-9

    # 1. Handle r_sat being zero (should not happen for valid orbital data)
    norm_r = np.linalg.norm(r_sat)
    if norm_r < epsilon:
        # Position vector is zero, orientation is undefined. Return identity.
        return np.array([1.0, 0.0, 0.0, 0.0])
        
    r_unit = r_sat / norm_r
    z_axis_body = -r_unit  # Body Z-axis (nadir)

    # 2. Determine Body X-axis and Y-axis
    norm_v = np.linalg.norm(v_sat)
    
    use_conventional_x = False
    if norm_v < epsilon:
        use_conventional_x = True # Velocity is zero or not provided
    else:
        v_unit = v_sat / norm_v
        # Check if velocity is radial (collinear with position)
        if abs(np.dot(v_unit, r_unit)) > (1.0 - epsilon):
            use_conventional_x = True # Velocity is radial
        else:
            # Velocity is available and not radial, use it for X-axis
            x_axis_body = v_unit
            
            y_axis_body_candidate = np.cross(z_axis_body, x_axis_body)
            norm_y_candidate = np.linalg.norm(y_axis_body_candidate)
            if norm_y_candidate < epsilon:
                # This case (X_body collinear with Z_body) should have been caught by radial check.
                # Fallback to conventional X if something unexpected happens.
                use_conventional_x = True
            else:
                y_axis_body = y_axis_body_candidate / norm_y_candidate

    if use_conventional_x:
        # Define X-axis conventionally when velocity is unusable.
        # Use EME2000 Z-axis ([0,0,1], inertial North) as a primary reference.
        ref_vector_inertial_z = np.array([0.0, 0.0, 1.0])
        
        x_axis_candidate = np.cross(ref_vector_inertial_z, z_axis_body)
        
        if np.linalg.norm(x_axis_candidate) < epsilon:
            # z_axis_body is aligned or anti-aligned with ref_vector_inertial_z
            # (i.e., satellite is near an inertial pole, z_body is along +/- inertial Z).
            # Pick X_body to align with inertial X (e.g., towards Vernal Equinox).
            x_axis_body = normalize(np.array([1.0, 0.0, 0.0])) # normalize() is safe for unit vectors
        else:
            x_axis_body = normalize(x_axis_candidate) # normalize() is safe here due to norm check
            
        y_axis_body = normalize(np.cross(z_axis_body, x_axis_body)) # Z and X are ortho-normal

    # 3. Construct Rotation Matrix (Inertial to Body)
    # Columns of R are the body axes expressed in the inertial frame.
    R = np.array([x_axis_body, y_axis_body, z_axis_body]).T

    # 4. Convert Rotation Matrix to Quaternion (scalar first: q0, q1, q2, q3)
    # Using a robust method (Sheppard's method adapted for stability).
    q = np.zeros(4)
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0.0: # Preferred case for numerical stability
        S = np.sqrt(trace + 1.0) * 2.0
        q[0] = 0.25 * S  # qw (scalar part)
        q[1] = (R[2, 1] - R[1, 2]) / S  # qx
        q[2] = (R[0, 2] - R[2, 0]) / S  # qy
        q[3] = (R[1, 0] - R[0, 1]) / S  # qz
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]): # R[0,0] is largest diagonal element
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q[0] = (R[2, 1] - R[1, 2]) / S
        q[1] = 0.25 * S
        q[2] = (R[0, 1] + R[1, 0]) / S
        q[3] = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]: # R[1,1] is largest diagonal element
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q[0] = (R[0, 2] - R[2, 0]) / S
        q[1] = (R[0, 1] + R[1, 0]) / S
        q[2] = 0.25 * S
        q[3] = (R[1, 2] + R[2, 1]) / S
    else: # R[2,2] is largest diagonal element
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q[0] = (R[1, 0] - R[0, 1]) / S
        q[1] = (R[0, 2] + R[2, 0]) / S
        q[2] = (R[1, 2] + R[2, 1]) / S
        q[3] = 0.25 * S
        
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

def parse_ccsds_oem_file(file_path):
    """Parse a CCSDS OEM (Orbit Ephemeris Message) file."""
    metadata = {"OBJECT_NAME": "UNKNOWN", "OBJECT_ID": "UNKNOWN"}
    state_vectors = []
    parsing_metadata = True
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("COMMENT"):
                    continue

                if parsing_metadata:
                    if "=" in line:
                        key, value = [x.strip() for x in line.split("=", 1)]
                        if key == "OBJECT_NAME":
                            metadata["OBJECT_NAME"] = value
                        elif key == "OBJECT_ID":
                            metadata["OBJECT_ID"] = value
                    if line == "META_STOP":
                        parsing_metadata = False
                    continue
                
                # Data lines
                parts = line.split()
                if len(parts) >= 5: # Expect at least MJD_day, MJD_seconds, X, Y, Z. Velocities are optional.
                    try:
                        # OEM data lines: MJD_day MJD_seconds_of_day X Y Z [VX VY VZ]
                        # The first two columns are interpreted as MJD day and seconds of that day,
                        mjd_day = int(parts[0])
                        mjd_seconds_of_day = float(parts[1])
                        
                        pos = np.array([float(parts[2]), float(parts[3]), float(parts[4])]) # Position in km
                        if len(parts) >= 8:
                            vel = np.array([float(parts[5]), float(parts[6]), float(parts[7])]) # Velocity in km/s
                        else:
                            # Velocities are optional, defaulting to [0,0,0] if not provided
                            vel = np.array([0.0, 0.0, 0.0])
                        state_vectors.append(((mjd_day, mjd_seconds_of_day), pos, vel))
                    except ValueError as e:
                        print(f"Skipping malformed data line: {line} - {e}")
                        continue
    except FileNotFoundError:
        raise Exception(f"Position file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error parsing position file: {e}")
    
    if not state_vectors and not parsing_metadata: # Ensure META_STOP was found if data is expected
        raise Exception("No state vectors found after META_STOP in position file, or META_STOP missing.")
    if not state_vectors and parsing_metadata: # META_STOP never found
        raise Exception("META_STOP not found in position file.")

    return metadata, state_vectors

def generate_quaternions_from_oem_vectors(oem_state_vectors):
    """Generate quaternion data from parsed OEM state vectors."""
    quaternions_data = []
    if not oem_state_vectors:
        return quaternions_data

    for (mjd_day, mjd_seconds_of_day), r_sat, v_sat in oem_state_vectors:
        # calculate_quaternion expects r_sat, v_sat in km and km/s, which OEM provides.
        quaternion = calculate_quaternion(r_sat, v_sat)
        # The output format requires MJD day and MJD seconds of day.
        quaternions_data.append((mjd_day, mjd_seconds_of_day, quaternion))
    
    return quaternions_data

# GUI Implementation
class TLE2nadirApp:
    def __init__(self, root, start_date="2025-01-13 00:00:00", end_date="2025-01-20 00:00:00", sampling=60, output_file="quaternions.txt", satellite_id=None):
        self.root = root
        self.root.title("TLE2nadir Generator")

        # Store the project directory from the output file path argument
        self.project_directory = os.path.dirname(output_file) if output_file else "."

        # Create Tab Control
        self.notebook = ttk.Notebook(root)
        
        self.tab_tle = ttk.Frame(self.notebook)
        self.tab_position_file = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_tle, text='TLE Input')
        self.notebook.add(self.tab_position_file, text='Position File Input')
        
        self.notebook.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)

        # --- TLE Input Tab ---
        # Satellite ID Input
        tk.Label(self.tab_tle, text="Satellite NORAD ID:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.norad_id = tk.StringVar(value=satellite_id)
        tk.Entry(self.tab_tle, textvariable=self.norad_id, width=10).grid(row=0, column=1, sticky="w", padx=5, pady=5)
        self.norad_name = tk.StringVar()
        tk.Entry(self.tab_tle, textvariable=self.norad_name, width=35, state="disabled").grid(row=0, column=1, sticky="e", padx=5, pady=5) # Adjusted column span or position
        tk.Button(self.tab_tle, text="Download", command=self.download_tle).grid(row=0, column=2, sticky="w", padx=5, pady=5)

        # TLE Input Fields
        tk.Label(self.tab_tle, text="TLE data:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.tle1 = tk.StringVar()
        self.tle2 = tk.StringVar()
        self.tle1_field = tk.Entry(self.tab_tle, textvariable=self.tle1, width=65)
        self.tle1_field.grid(row=1, column=1, columnspan=2, sticky="w", padx=5, pady=2)
        self.tle1_field.bind("<<Paste>>", self.handle_tle_paste)
        self.tle2_field = tk.Entry(self.tab_tle, textvariable=self.tle2, width=65)
        self.tle2_field.grid(row=2, column=1, columnspan=2, sticky="w", padx=5, pady=2)
 
        # Start/End Date Inputs
        tk.Label(self.tab_tle, text="Start date:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.start_time = tk.StringVar(value=start_date)
        tk.Entry(self.tab_tle, textvariable=self.start_time, width=30).grid(row=3, column=1, columnspan=2, sticky="w", padx=5, pady=5) # Adjusted columnspan

        tk.Label(self.tab_tle, text="End date:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.end_time = tk.StringVar(value=end_date)
        tk.Entry(self.tab_tle, textvariable=self.end_time, width=30).grid(row=4, column=1, columnspan=2, sticky="w", padx=5, pady=5) # Adjusted columnspan

        # Sampling Interval
        tk.Label(self.tab_tle, text="Sampling interval:").grid(row=5, column=0, sticky="w", padx=5, pady=5)
        self.interval = tk.StringVar(value=sampling)
        tk.Entry(self.tab_tle, textvariable=self.interval, width=10).grid(row=5, column=1, sticky="w", padx=5, pady=5)
        tk.Label(self.tab_tle, text="seconds").grid(row=5, column=2, sticky="w", padx=5, pady=5)

        # --- Position File Input Tab ---
        tk.Label(self.tab_position_file, text="Position File (CCSDS OEM):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.position_file_path = tk.StringVar()
        tk.Entry(self.tab_position_file, textvariable=self.position_file_path, width=50).grid(row=0, column=1, sticky="w", padx=5, pady=5)
        tk.Button(self.tab_position_file, text="Browse", command=self.browse_position_file).grid(row=0, column=2, sticky="w", padx=5, pady=5)
        
        # --- Common Controls (Output File & Generate Button) ---
        # These are placed on the main root window, below the notebook
        common_controls_frame = tk.Frame(root)
        common_controls_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

        tk.Label(common_controls_frame, text="Output file:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.output_file = tk.StringVar(value=output_file)
        tk.Entry(common_controls_frame, textvariable=self.output_file, width=50).grid(row=0, column=1, sticky="w", padx=5, pady=5)
        tk.Button(common_controls_frame, text="Browse", command=self.browse_output_file).grid(row=0, column=2, sticky="w", padx=5, pady=5)

        tk.Button(common_controls_frame, text="Generate", command=self.generate_quaternions, bg="green", fg="white").grid(row=1, column=1, pady=10)

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
            self.norad_name.set(tle[0])
            self.tle1.set(tle[1])
            self.tle2.set(tle[2])
        except Exception as e:
            messagebox.showerror("Error", f"Could not download TLE: {e}")

    def browse_output_file(self):
        """Select output file path."""
        file_path = filedialog.asksaveasfilename(initialdir=self.project_directory, defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
        if file_path:
            self.output_file.set(file_path)

    def browse_position_file(self):
        """Select position file path."""
        file_path = filedialog.askopenfilename(initialdir=self.project_directory, title="Select Position File", filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if file_path:
            self.position_file_path.set(file_path)

    def _get_input_data_for_generation(self):
        """
        Prepares and returns the data needed for quaternion generation based on the selected input tab.
        Returns:
            tuple: (quaternion_data_list, object_name, object_id) or (None, None, None) if error.
        """
        quaternion_data_list = None
        object_name = "UNKNOWN"
        object_id = "UNKNOWN"

        try:
            selected_tab_index = self.notebook.index(self.notebook.select())

            if selected_tab_index == 1:  # Position File Tab
                pos_file_path = self.position_file_path.get()
                if not pos_file_path:
                    messagebox.showerror("Error", "Please select a position file.")
                    return None, None, None
                
                metadata, oem_state_vectors = parse_ccsds_oem_file(pos_file_path)
                quaternion_data_list = generate_quaternions_from_oem_vectors(oem_state_vectors)
                object_name = metadata.get("OBJECT_NAME", "UNKNOWN")
                object_id = metadata.get("OBJECT_ID", "UNKNOWN")
                
                if not quaternion_data_list: # Parsing OK, but no data vectors
                    messagebox.showwarning("Warning", "No state vectors found in the position file to generate quaternions.")
                    return [], object_name, object_id # Return empty list, not an error state


            elif selected_tab_index == 0:  # TLE Input Tab
                if not self.tle1.get() or not self.tle2.get():
                    messagebox.showerror("Error", "TLE data is required for TLE input method.")
                    return None, None, None
                
                start_time_str = self.start_time.get()
                end_time_str = self.end_time.get()
                interval_str = self.interval.get()

                if not all([start_time_str, end_time_str, interval_str]):
                    messagebox.showerror("Error", "Start date, End date, and Sampling interval are required for TLE input.")
                    return None, None, None

                start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
                end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")
                interval = int(interval_str)
                
                ts = load.timescale()
                satellite = EarthSatellite(self.tle1.get(), self.tle2.get())
                quaternion_data_list = generate_quaternions_over_time(satellite, ts, start_time, end_time, interval)
                object_name = self.norad_name.get() or "UNKNOWN"
                object_id = self.norad_id.get() or "UNKNOWN"

                if not quaternion_data_list:
                    messagebox.showwarning("Warning", "No quaternions generated from TLE data. Check time range and TLE.")
                    return [], object_name, object_id # Return empty list
            else:
                # Should not be reached with proper tab setup
                messagebox.showerror("Error", "Please select a valid input tab (TLE or Position File).")
                return None, None, None
                
        except ValueError as ve: # strptime and int conversion errors
            messagebox.showerror("Error", f"Invalid date format or interval: {ve}")
            return None, None, None
        except Exception as e: # File parsing, Skyfield, or other data prep errors
            messagebox.showerror("Error", f"Failed to prepare data: {e}")
            return None, None, None

        return quaternion_data_list, object_name, object_id

    def generate_quaternions(self):
        """Generate quaternions and save to file."""
        try:
            output_path = self.output_file.get()
            if not output_path:
                messagebox.showerror("Error", "Output file path cannot be empty.")
                return

            quaternion_data_list, object_name_for_header, object_id_for_header = self._get_input_data_for_generation()

            # Critical input errors handled by _get_input_data_for_generation (returns None)
            if quaternion_data_list is None:
                return # Error message already shown
            
            # Data prep OK, but no actual data points (e.g., empty OEM after META_STOP)
            if not quaternion_data_list:
                # Helper method (_get_input_data_for_generation) shows specific warnings for empty data.
                # A general info message here confirms no file will be written.
                messagebox.showinfo("Info", "No data points were generated to write to the output file.")
                return

            with open(output_path, 'w') as f:
                f.write(f"""CIC_AEM_VERS = 1.0
COMMENT Generated by yasiu.pl TLE2nadir generator (https://github.com/yasiupl/TLE2nadir)
CREATION_DATE = {(datetime.now(timezone.utc)).isoformat(timespec='microseconds')}
ORIGINATOR = VTS

META_START

OBJECT_NAME = {object_name_for_header}
OBJECT_ID = {object_id_for_header}
REF_FRAME_A = EME2000
REF_FRAME_B = UNDEFINED
ATTITUDE_DIR = A2B
TIME_SYSTEM = UTC
ATTITUDE_TYPE = QUATERNION

META_STOP
\n
""")
                for days, seconds, quaternion_data in quaternion_data_list:
                    f.write(generate_ccsds_quaternion_data(days, seconds, quaternion_data) + '\n')

            messagebox.showinfo("Success", "Quaternion data saved successfully.")
        except Exception as e: # Catch any unexpected errors during file writing or final steps
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
