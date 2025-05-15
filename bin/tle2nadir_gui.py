"""
VTS Timeloop Generator plugin for Nadir-Pointing Quaternions.

This script provides a GUI application to generate nadir-pointing quaternions
for a satellite based on its Two-Line Element set (TLE) or a CCSDS Orbit
Ephemeris Message (OEM) file over a specified time period.

The output is a CCSDS Attitude Ephemeris Message (AEM) formatted file.

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
import argparse
import logging
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Tuple, Optional, List, Any # For type hinting
from datetime import datetime, timedelta, timezone

import requests
from skyfield.api import EarthSatellite, load

import core_logic # Changed from relative to direct import

__author__ = "Marcin Jasiukowicz"
__copyright__ = "Copyright (C) 2025  Marcin Jasiukowicz"
__license__ = "GPLv3"
__version__ = "1.0.0"
__email__ = "contact@yasiu.pl"
__status__ = "Production"

# Module-level constants
CELESTRAK_GP_URL_FORMAT = "https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=TLE"


def _parse_mjd_pair(value: str) -> Tuple[int, float]:
    """
    Custom argparse type for a pair of MJD day (INT) and MJD seconds (FLOAT).
    Intended for internal use by argparse.

    Example input string: "60688 38356.397000"

    Args:
        value: The input string from the command line.

    Returns:
        A tuple containing (MJD_days, MJD_seconds_of_day).

    Raises:
        argparse.ArgumentTypeError: If the value cannot be parsed as two numbers.
    """
    try:
        mjd_days, mjd_seconds = value.split()
        return int(mjd_days), float(mjd_seconds)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"'{value}' is not a valid pair of INT and FLOAT. Example: '60688 38356.397000'"
        )

def _mjd_to_datetime(mjd: float) -> datetime:
    """
    Convert a Modified Julian Date (MJD) to a naive datetime object (UTC based, no timezone info).
    Fractional seconds from MJD are truncated.

    Args:
        mjd: Modified Julian Date.

    Returns:
        A naive datetime object corresponding to the MJD.
    """
    mjd_epoch_dt = datetime(1858, 11, 17, tzinfo=timezone.utc)  # MJD epoch datetime
    total_seconds_from_epoch = int(mjd * 86400.0)  # Convert MJD to total seconds and truncate
    utc_dt = mjd_epoch_dt + timedelta(seconds=total_seconds_from_epoch)
    return utc_dt.replace(tzinfo=None)  # Return as naive datetime


# Utility function for TLE fetching
def _fetch_tle_from_celestrak(norad_id_str: str) -> Tuple[str, str, str]:
    """
    Fetches TLE data from Celestrak for a given NORAD ID.

    Args:
        norad_id_str: The NORAD ID as a string.

    Returns:
        A tuple containing (satellite_name, tle_line1, tle_line2).

    Raises:
        requests.exceptions.RequestException: If a network error occurs.
        ValueError: If the downloaded TLE data is not in the expected format.
    """
    logging.info(f"Fetching TLE for NORAD ID: {norad_id_str} from Celestrak.")
    url = CELESTRAK_GP_URL_FORMAT.format(norad_id=norad_id_str)
    response = requests.get(url, timeout=10)
    response.raise_for_status()  # Raise HTTPError for bad responses (4XX or 5XX)
    
    tle_data = response.text.strip().split('\n')
    if len(tle_data) >= 3:
        sat_name = tle_data[0].strip()
        line1 = tle_data[1].strip()
        line2 = tle_data[2].strip()
        return sat_name, line1, line2
    else:
        raise ValueError("Downloaded TLE data is not in the expected format (Name, Line1, Line2).")

# GUI Implementation
class TLE2nadirApp:
    """
    Main application class for the TLE2nadir Tkinter GUI.

    This class sets up the UI, handles user interactions, and orchestrates
    the quaternion generation process by calling functions from `core_logic`.
    """

    def __init__(self,
                 root: tk.Tk,
                 start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None,
                 sampling: int = 60,
                 output_file: str = "quaternions.txt",
                 satellite_id: Optional[str] = None):
        """
        Initialize the TLE2nadirApp GUI.

        Args:
            root: The root Tkinter window.
            start_date: Default start datetime for TLE processing.
                        If None, a default string "2025-01-13 00:00:00" is used.
            end_date: Default end datetime for TLE processing.
                      If None, a default string "2025-01-20 00:00:00" is used.
            sampling: Default sampling interval in seconds.
            output_file: Default output file path.
            satellite_id: Default NORAD ID for TLE download.
        """
        self.root = root
        self.root.title("TLE2nadir Generator")

        # Use provided dates or fallback to string defaults for display
        start_date_str = start_date.strftime("%Y-%m-%d %H:%M:%S") if start_date else "2025-01-13 00:00:00"
        end_date_str = end_date.strftime("%Y-%m-%d %H:%M:%S") if end_date else "2025-01-20 00:00:00"

        self.project_directory: str = os.path.dirname(output_file) if output_file else "."

        # Create Tab Control
        self.notebook = ttk.Notebook(root)
        
        self.tab_tle = ttk.Frame(self.notebook)
        self.tab_position_file = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_tle, text='TLE Input')
        self.notebook.add(self.tab_position_file, text='Position File Input')
        
        self.notebook.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)

        # --- TLE Input Tab ---
        tk.Label(self.tab_tle, text="Satellite NORAD ID:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.norad_id = tk.StringVar(value=satellite_id if satellite_id else "")
        tk.Entry(self.tab_tle, textvariable=self.norad_id, width=10).grid(row=0, column=1, sticky="w", padx=5, pady=5)
        self.norad_name = tk.StringVar()
        tk.Entry(self.tab_tle, textvariable=self.norad_name, width=35, state="disabled").grid(row=0, column=1, sticky="e", padx=5, pady=5)
        tk.Button(self.tab_tle, text="Download", command=self.download_tle).grid(row=0, column=2, sticky="w", padx=5, pady=5)

        tk.Label(self.tab_tle, text="TLE data:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.tle1 = tk.StringVar()
        self.tle2 = tk.StringVar()
        self.tle1_field = tk.Entry(self.tab_tle, textvariable=self.tle1, width=65)
        self.tle1_field.grid(row=1, column=1, columnspan=2, sticky="w", padx=5, pady=2)
        self.tle1_field.bind("<<Paste>>", self.handle_tle_paste)
        self.tle2_field = tk.Entry(self.tab_tle, textvariable=self.tle2, width=65)
        self.tle2_field.grid(row=2, column=1, columnspan=2, sticky="w", padx=5, pady=2)
 
        tk.Label(self.tab_tle, text="Start date:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.start_time = tk.StringVar(value=start_date_str)
        tk.Entry(self.tab_tle, textvariable=self.start_time, width=30).grid(row=3, column=1, columnspan=2, sticky="w", padx=5, pady=5)

        tk.Label(self.tab_tle, text="End date:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.end_time = tk.StringVar(value=end_date_str)
        tk.Entry(self.tab_tle, textvariable=self.end_time, width=30).grid(row=4, column=1, columnspan=2, sticky="w", padx=5, pady=5)

        tk.Label(self.tab_tle, text="Sampling interval:").grid(row=5, column=0, sticky="w", padx=5, pady=5)
        self.interval = tk.StringVar(value=str(sampling))
        tk.Entry(self.tab_tle, textvariable=self.interval, width=10).grid(row=5, column=1, sticky="w", padx=5, pady=5)
        tk.Label(self.tab_tle, text="seconds").grid(row=5, column=2, sticky="w", padx=5, pady=5)

        # --- Position File Input Tab ---
        tk.Label(self.tab_position_file, text="Position File (CCSDS OEM):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.position_file_path = tk.StringVar()
        tk.Entry(self.tab_position_file, textvariable=self.position_file_path, width=50).grid(row=0, column=1, sticky="w", padx=5, pady=5)
        tk.Button(self.tab_position_file, text="Browse", command=self.browse_position_file).grid(row=0, column=2, sticky="w", padx=5, pady=5)
        
        # --- Common Controls (Output File & Generate Button) ---
        common_controls_frame = tk.Frame(root)
        common_controls_frame.grid(row=1, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

        tk.Label(common_controls_frame, text="Output file:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.output_file = tk.StringVar(value=output_file)
        tk.Entry(common_controls_frame, textvariable=self.output_file, width=50).grid(row=0, column=1, sticky="w", padx=5, pady=5)
        tk.Button(common_controls_frame, text="Browse", command=self.browse_output_file).grid(row=0, column=2, sticky="w", padx=5, pady=5)

        tk.Button(common_controls_frame, text="Generate", command=self.generate_quaternions, bg="green", fg="white").grid(row=1, column=1, pady=10)

    def handle_tle_paste(self, event: tk.Event) -> None:
        """
        Handle paste events (Ctrl+V or context menu paste) for the TLE Line 1 field.
        
        If the pasted content contains multiple lines, it attempts to populate
        TLE Line 1 and TLE Line 2 fields automatically.

        Args:
            event: The Tkinter event object (unused in this method but required by bind).
        """
        try:
            clipboard_content = self.root.clipboard_get()
            tle_lines = clipboard_content.split("\n")

            self.tle1_field.delete(0, tk.END)
            self.tle1_field.insert(0, tle_lines[0].strip())

            if len(tle_lines) > 1:
                self.tle2_field.delete(0, tk.END)
                self.tle2_field.insert(0, tle_lines[1].strip())
        except tk.TclError:
            # This can happen if clipboard content is not text (e.g., an image)
            logging.warning("Failed to paste TLE data; clipboard content might not be text.", exc_info=True)
            # Optionally, show a gentle error to the user, or just pass silently.
            # messagebox.showwarning("Paste Error", "Could not paste TLE data. Please ensure you are pasting valid text.")
            pass


    def download_tle(self) -> None:
        """
        Handles the TLE download process initiated by the user.
        Validates NORAD ID, calls helper to fetch TLE, and updates GUI.
        """
        norad_id_str = self.norad_id.get()
        if not norad_id_str.isdigit():
            err_msg = "Please enter a valid NORAD ID (numbers only)."
            logging.error(f"Download TLE Error: {err_msg} (Provided ID: '{norad_id_str}')")
            messagebox.showerror("Error", err_msg)
            return

        try:
            sat_name, line1, line2 = _fetch_tle_from_celestrak(norad_id_str)
            self.norad_name.set(sat_name)
            self.tle1.set(line1)
            self.tle2.set(line2)
            logging.info(f"Successfully downloaded and displayed TLE for '{sat_name}' (NORAD ID: {norad_id_str}).")
        except requests.exceptions.RequestException as e:
            err_msg = f"Could not download TLE (network error): {e}"
            logging.error(f"Download TLE RequestException for NORAD ID {norad_id_str}: {err_msg}", exc_info=True)
            messagebox.showerror("Error", err_msg)
        except ValueError as e: # Catch format error from _fetch_tle_from_celestrak
            err_msg = f"Error processing downloaded TLE: {e}"
            logging.error(f"{err_msg} (NORAD ID: {norad_id_str})", exc_info=True)
            messagebox.showerror("Error", err_msg)
        except Exception as e:  # Catch any other unexpected errors
            err_msg = f"An unexpected error occurred while downloading TLE: {e}"
            logging.error(f"Download TLE Exception for NORAD ID {norad_id_str}: {err_msg}", exc_info=True)
            messagebox.showerror("Error", err_msg)


    def browse_output_file(self) -> None:
        """Open a 'Save As' dialog to select the output file path."""
        file_path = filedialog.asksaveasfilename(
            initialdir=self.project_directory,
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if file_path:
            self.output_file.set(file_path)
            logging.info(f"Output file path set to: {file_path}")


    def browse_position_file(self) -> None:
        """Open an 'Open File' dialog to select a position file (e.g., CCSDS OEM)."""
        file_path = filedialog.askopenfilename(
            initialdir=self.project_directory,
            title="Select Position File (e.g., CCSDS OEM)",
            filetypes=[("Text Files", "*.txt"), ("OEM Files", "*.oem"), ("All Files", "*.*")]
        )
        if file_path:
            self.position_file_path.set(file_path)
            logging.info(f"Position file path set to: {file_path}")


    def _prepare_data_from_tle_input(self) -> Optional[Tuple[List[Any], str, str]]:
        """
        Handles data preparation if the TLE input tab is selected.
        Validates inputs, parses dates, and calls core logic to generate quaternions.

        Returns:
            A tuple (quaternion_data_list, object_name, object_id) on success.
            Returns ([], object_name, object_id) if no quaternions are generated but input is valid.
            Returns None if there's a critical input error.
        """
        if not self.tle1.get() or not self.tle2.get():
            err_msg = "TLE data (Line 1 and Line 2) is required for TLE input method."
            logging.error(err_msg)
            messagebox.showerror("Error", err_msg)
            return None
        
        start_time_str = self.start_time.get()
        end_time_str = self.end_time.get()
        interval_str = self.interval.get()

        if not all([start_time_str, end_time_str, interval_str]):
            err_msg = "Start date, End date, and Sampling interval are required for TLE input."
            logging.error(err_msg)
            messagebox.showerror("Error", err_msg)
            return None

        try:
            start_dt = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
            end_dt = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")
            interval_sec = int(interval_str)
            if interval_sec <= 0:
                raise ValueError("Sampling interval must be a positive integer.")
            if start_dt > end_dt:
                raise ValueError("Start date cannot be after end date.")
        except ValueError as ve:
            err_msg = f"Invalid date, time, or interval for TLE input: {ve}"
            logging.error(err_msg, exc_info=True)
            messagebox.showerror("Error", err_msg)
            return None

        logging.info(f"Preparing data from TLE input: NORAD ID {self.norad_id.get()}, "
                     f"Start: {start_dt}, End: {end_dt}, Interval: {interval_sec}s")
        ts = load.timescale()
        try:
            satellite = EarthSatellite(self.tle1.get(), self.tle2.get(), self.norad_name.get() or "TLE Satellite")
        except ValueError as e: # Skyfield can raise ValueError for malformed TLEs
            err_msg = f"Invalid TLE data: {e}"
            logging.error(err_msg, exc_info=True)
            messagebox.showerror("Error", err_msg)
            return None

        quaternion_data_list = core_logic.generate_quaternions_over_period(
            satellite, ts, start_dt, end_dt, interval_sec
        )
        
        object_name = self.norad_name.get() or "UNKNOWN_TLE_SAT"
        object_id = self.norad_id.get() or "N/A" # NORAD ID is the primary ID here

        if not quaternion_data_list:
            warn_msg = "No quaternions generated from TLE data. Check time range and TLE validity."
            logging.warning(warn_msg)
            messagebox.showwarning("Warning", warn_msg)
            return [], object_name, object_id
        
        logging.info("Successfully prepared data from TLE input.")
        return quaternion_data_list, object_name, object_id


    def _prepare_data_from_position_file(self) -> Optional[Tuple[List[Any], str, str]]:
        """
        Handles data preparation if the Position File input tab is selected.
        Validates input and calls core logic to parse the file and generate quaternions.

        Returns:
            A tuple (quaternion_data_list, object_name, object_id) on success.
            Returns ([], object_name, object_id) if no quaternions are generated but input is valid.
            Returns None if there's a critical input error.
        """
        pos_file_path = self.position_file_path.get()
        if not pos_file_path:
            err_msg = "Please select a position file (e.g., CCSDS OEM)."
            logging.error(err_msg)
            messagebox.showerror("Error", err_msg)
            return None
        
        logging.info(f"Preparing data from position file: {pos_file_path}")
        # Errors from core_logic.parse_ccsds_oem_file will be caught by the caller
        metadata, oem_state_vectors = core_logic.parse_ccsds_oem_file(pos_file_path)
        quaternion_data_list = core_logic.generate_quaternions_from_oem_vectors(oem_state_vectors)
        
        object_name = metadata.get("OBJECT_NAME", "UNKNOWN_OEM_SAT")
        object_id = metadata.get("OBJECT_ID", "N/A") # OBJECT_ID from OEM is primary
        
        if not quaternion_data_list and oem_state_vectors: # Parsed vectors but all failed quaternion gen (unlikely with current logic)
             warn_msg = "State vectors found, but no quaternions could be generated."
             logging.warning(f"{warn_msg} (File: {pos_file_path})")
             messagebox.showwarning("Warning", warn_msg)
             return [], object_name, object_id
        elif not quaternion_data_list and not oem_state_vectors: # No vectors found after parsing
            warn_msg = "No state vectors found in the position file to generate quaternions."
            logging.warning(f"{warn_msg} (File: {pos_file_path})")
            messagebox.showwarning("Warning", warn_msg)
            return [], object_name, object_id
            
        logging.info(f"Successfully prepared data from position file: {pos_file_path}")
        return quaternion_data_list, object_name, object_id


    def _get_input_data_for_generation(self) -> Optional[Tuple[List[Any], str, str]]:
        """
        Orchestrates data preparation based on the currently selected input tab.

        Calls helper methods `_prepare_data_from_tle_input` or
        `_prepare_data_from_position_file`.

        Returns:
            A tuple (quaternion_data_list, object_name, object_id) on success.
            Returns ([], object_name, object_id) if no quaternions are generated but input is valid.
            Returns None if there's a critical input error during data preparation.
        """
        try:
            selected_tab_index = self.notebook.index(self.notebook.select())

            if selected_tab_index == 1:  # Position File Tab
                return self._prepare_data_from_position_file()
            elif selected_tab_index == 0:  # TLE Input Tab
                return self._prepare_data_from_tle_input()
            else:
                err_msg = "Invalid input tab selected. This should not happen."
                logging.error(err_msg)
                messagebox.showerror("Error", err_msg)
                return None
                
        except Exception as e: # Catch-all for unexpected errors during data prep dispatch
            err_msg = f"Failed to prepare data due to an unexpected error: {e}"
            logging.error(err_msg, exc_info=True)
            messagebox.showerror("Error", err_msg)
            return None


    def generate_quaternions(self) -> None:
        """
        Main function to generate quaternions and save them to the specified output file.
        
        It retrieves input data using `_get_input_data_for_generation`, then formats
        and writes the output CCSDS AEM file.
        """
        try:
            output_path = self.output_file.get()
            if not output_path:
                err_msg = "Output file path cannot be empty."
                logging.error(err_msg)
                messagebox.showerror("Error", err_msg)
                return

            logging.info("Starting quaternion generation process.")
            result = self._get_input_data_for_generation()

            if result is None:
                # Error already logged and shown by _get_input_data_for_generation or its sub-methods
                logging.warning("Quaternion generation aborted due to data preparation errors.")
                return
            
            quaternion_data_list, object_name_for_header, object_id_for_header = result
            
            if not quaternion_data_list:
                info_msg = "No data points were generated to write to the output file."
                logging.info(info_msg) # Already logged by sub-methods, but good for overall process trace
                messagebox.showinfo("Info", info_msg) # Messagebox also shown by sub-methods
                return

            logging.info(f"Attempting to write quaternion data to: {output_path} for {object_name_for_header} ({object_id_for_header})")
            
            # Call core_logic to write the file
            core_logic.write_aem_file(
                output_path=output_path,
                data_lines=quaternion_data_list,
                object_name=object_name_for_header,
                object_id=object_id_for_header,
                originator="VTS", # Or make this configurable if needed
                generator_comment="Generated by yasiu.pl TLE2nadir generator (https://github.com/yasiupl/TLE2nadir)"
            )

            success_msg = f"Quaternion data saved successfully to {output_path}"
            logging.info(success_msg)
            messagebox.showinfo("Success", success_msg)
        except Exception as e:
            err_msg = f"Failed to generate or write quaternions: {e}"
            logging.error(err_msg, exc_info=True)
            messagebox.showerror("Error", err_msg)


def main() -> None:
    """
    Main entry point for the TLE2nadir application.

    Configures logging, parses command-line arguments (if any, for VTS integration),
    and launches the Tkinter GUI.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__) # Get a logger for this module

    parser = argparse.ArgumentParser(
        description='GUI for generating Nadir-pointing quaternions for VTS Timeloop.'
    )

    parser.add_argument('--vtsgentype',
                        choices=['GenPosition', 'GenQuaternion', 'GenEuler', 'GenAxis',
                                 'GenAngle', 'GenDirection', 'GenAltAz'],
                        help='VTS Generator type (typically GenQuaternion for this tool).')
    parser.add_argument('--vtsdefaultfile', help='VTS default output file path.')
    parser.add_argument('--vtsmjddates', nargs=2, type=_parse_mjd_pair,
                        metavar=('StartMJD', 'EndMJD'),
                        help='VTS Start and End dates as MJD day and MJD seconds (e.g., "day1 secs1" "day2 secs2").')
    # Add a debug flag
    parser.add_argument('--debug', action='store_true', help='Enable debug logging level.')


    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG) # Set root logger to DEBUG
        logger.debug("Debug logging enabled.")


    app_start_date: Optional[datetime] = None
    app_end_date: Optional[datetime] = None
    app_output_file: str = "quaternions.txt" # Default if not from VTS

    if args.vtsgentype:
        logger.info(f"Launched by VTS with type: {args.vtsgentype}")
        if args.vtsgentype != 'GenQuaternion':
            err_msg = f"Operation type '{args.vtsgentype}' is not supported by this TLE2nadir GUI. Expected 'GenQuaternion'."
            logger.error(err_msg)
            # For VTS, messagebox might not be ideal if it's a non-interactive launch.
            # Consider if VTS handles stderr or if a specific exit code is better.
            messagebox.showerror("Unsupported VTS Type", err_msg)
            return # Exit if launched with unsupported type by VTS

        if args.vtsdefaultfile:
            app_output_file = args.vtsdefaultfile
            logger.info(f"VTS default output file: {app_output_file}")

        if args.vtsmjddates:
            try:
                start_mjd_days, start_mjd_seconds = args.vtsmjddates[0]
                end_mjd_days, end_mjd_seconds = args.vtsmjddates[1]
                
                app_start_date = _mjd_to_datetime(start_mjd_days + start_mjd_seconds / 86400.0)
                app_end_date = _mjd_to_datetime(end_mjd_days + end_mjd_seconds / 86400.0)
                logger.info(f"VTS MJD dates converted: Start={app_start_date}, End={app_end_date}")
            except Exception as e:
                logger.error(f"Error converting VTS MJD dates: {e}", exc_info=True)
                messagebox.showerror("VTS Date Error", f"Could not parse MJD dates from VTS: {e}")
                # Decide if to proceed with defaults or exit
    else:
        logger.info("Launched as a standalone application.")


    root = tk.Tk()
    app = TLE2nadirApp(
        root,
        start_date=app_start_date, # Can be None, __init__ handles it
        end_date=app_end_date,     # Can be None, __init__ handles it
        output_file=app_output_file
        # satellite_id can be passed if there's a VTS way to specify it, or a config file
    )
    root.mainloop()


if __name__ == "__main__":
    main()
