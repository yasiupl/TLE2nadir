"""
Core logic for TLE2nadir quaternion generation.

This module handles the core calculations, data parsing, and transformations
required to generate nadir-pointing quaternions from satellite orbital data.
"""
import logging
from typing import Tuple, List, Dict, Any
import numpy as np
from datetime import datetime, timedelta, timezone
from skyfield.api import EarthSatellite, load, Timescale  # Keep skyfield imports here

EPSILON = 1e-9  # Epsilon for floating point comparisons


# Utility Functions
def _normalize(vector: np.ndarray) -> np.ndarray:
    """
    Normalize a 3D vector. (Internal use)

    Args:
        vector: The input vector (numpy array).

    Returns:
        The normalized vector. Returns a zero vector if the input norm is zero.
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector # Or raise an error, or return np.zeros_like(vector)
    return vector / norm


def _julian_date_to_mjd(jd: float) -> Tuple[int, float]:
    """
    Convert Julian Date (JD) to Modified Julian Date (MJD). (Internal use)

    Args:
        jd: Julian Date.

    Returns:
        A tuple containing MJD days (int) and MJD seconds of the day (float).
    """
    mjd_epoch = 2400000.5  # MJD epoch: 00:00 UTC on November 17, 1858
    delta_days_total = jd - mjd_epoch
    days = int(delta_days_total)
    seconds = (delta_days_total - days) * 86400.0
    return days, seconds

def _generate_ccsds_quaternion_data(days: int, seconds: float, quaternion: np.ndarray) -> str:
    """
    Generate a CCSDS-compatible quaternion string (Attitude Message Format). (Internal use)

    Format: MJD_days MJD_seconds_of_day q_scalar q_x q_y q_z

    Args:
        days: MJD days part.
        seconds: MJD seconds of the day part.
        quaternion: Numpy array representing the quaternion [qw, qx, qy, qz] (scalar first).

    Returns:
        A string formatted for CCSDS AEM quaternion data lines.
    """
    return f"{days}\t{seconds:.6f}\t{quaternion[0]:.6f}\t{quaternion[1]:.6f}\t{quaternion[2]:.6f}\t{quaternion[3]:.6f}"


# Main logic
def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Converts a 3x3 rotation matrix to a quaternion (scalar first: qw, qx, qy, qz).

    This implementation uses Sheppard's method, adapted for numerical stability by
    choosing the calculation path that avoids division by small numbers.

    Args:
        R: A 3x3 numpy array representing the rotation matrix.

    Returns:
        A 4-element numpy array representing the quaternion [qw, qx, qy, qz].
    """
    q = np.zeros(4)  # Initialize quaternion [qw, qx, qy, qz]
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0.0:  # Preferred case for numerical stability
        S = np.sqrt(trace + 1.0) * 2.0
        q[0] = 0.25 * S  # qw (scalar part)
        q[1] = (R[2, 1] - R[1, 2]) / S  # qx
        q[2] = (R[0, 2] - R[2, 0]) / S  # qy
        q[3] = (R[1, 0] - R[0, 1]) / S  # qz
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):  # R[0,0] is largest diagonal element
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q[0] = (R[2, 1] - R[1, 2]) / S
        q[1] = 0.25 * S
        q[2] = (R[0, 1] + R[1, 0]) / S
        q[3] = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:  # R[1,1] is largest diagonal element
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q[0] = (R[0, 2] - R[2, 0]) / S
        q[1] = (R[0, 1] + R[1, 0]) / S
        q[2] = 0.25 * S
        q[3] = (R[1, 2] + R[2, 1]) / S
    else:  # R[2,2] is largest diagonal element
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q[0] = (R[1, 0] - R[0, 1]) / S
        q[1] = (R[0, 2] + R[2, 0]) / S
        q[2] = (R[1, 2] + R[2, 1]) / S
        q[3] = 0.25 * S
    return q

def _determine_body_axes(r_unit: np.ndarray, v_sat: np.ndarray, z_axis_body: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Determines the satellite body X and Y axes for a nadir-pointing configuration.

    The body frame is defined as:
    - Z-axis (input `z_axis_body`): Typically points towards nadir (-r_unit).
    - X-axis: Aligns with the satellite's velocity vector projected onto the
              plane perpendicular to Z, if velocity is available and not radial.
              If velocity is unusable (zero, radial, or not provided), a
              conventional X-axis is chosen (e.g., using an inertial reference
              like EME2000 Z-axis to form a basis).
    - Y-axis: Completes the right-handed system (Y = Z x X).

    Args:
        r_unit: Normalized satellite position vector in the inertial frame.
        v_sat: Satellite velocity vector in the inertial frame.
        z_axis_body: Satellite body Z-axis vector in the inertial frame.

    Returns:
        A tuple containing the normalized X-axis and Y-axis of the body frame,
        expressed in the inertial frame.
    """
    norm_v = np.linalg.norm(v_sat)
    use_conventional_x = False

    if norm_v < EPSILON:
        use_conventional_x = True  # Velocity is zero or not provided
    else:
        v_unit = v_sat / norm_v
        # Check if velocity is radial (collinear with position)
        if abs(np.dot(v_unit, r_unit)) > (1.0 - EPSILON):
            use_conventional_x = True  # Velocity is radial
        else:
            # Velocity is available and not radial, use it for X-axis
            x_axis_body = v_unit
            y_axis_body_candidate = np.cross(z_axis_body, x_axis_body)
            norm_y_candidate = np.linalg.norm(y_axis_body_candidate)
            if norm_y_candidate < EPSILON:
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
        
        if np.linalg.norm(x_axis_candidate) < EPSILON:
            # z_axis_body is aligned or anti-aligned with ref_vector_inertial_z
            # (i.e., satellite is near an inertial pole, z_body is along +/- inertial Z).
            # Pick X_body to align with inertial X (e.g., towards Vernal Equinox).
            x_axis_body = _normalize(np.array([1.0, 0.0, 0.0]))  # Align with inertial X
        else:
            x_axis_body = _normalize(x_axis_candidate)
            
        y_axis_body = _normalize(np.cross(z_axis_body, x_axis_body))  # Ensure Y = Z x X
    
    return x_axis_body, y_axis_body


def _calculate_quaternion(r_sat: np.ndarray, v_sat: np.ndarray) -> np.ndarray:
    """
    Calculate the quaternion for a nadir-pointing satellite. (Internal use)

    This function determines the orientation of a satellite such that one axis
    (body Z) points towards the Earth's center (nadir), and another axis
    (body X) is defined based on the velocity vector or a conventional direction.
    The resulting orientation is expressed as a quaternion [qw, qx, qy, qz]
    representing the rotation from the EME2000 inertial frame to the satellite
    body frame.

    Args:
        r_sat: Satellite position vector in km (EME2000 frame).
        v_sat: Satellite velocity vector in km/s (EME2000 frame).

    Returns:
        A 4-element numpy array representing the quaternion [qw, qx, qy, qz]
        (scalar first). Returns an identity quaternion [1, 0, 0, 0] if the
        position vector is zero.
    """
    # 1. Handle r_sat being zero (should not happen for valid orbital data)
    norm_r = np.linalg.norm(r_sat)
    if norm_r < EPSILON:
        # Position vector is zero, orientation is undefined. Return identity.
        return np.array([1.0, 0.0, 0.0, 0.0])
        
    r_unit = r_sat / norm_r
    z_axis_body = -r_unit  # Body Z-axis (nadir)

    # 2. Determine Body X-axis and Y-axis
    x_axis_body, y_axis_body = _determine_body_axes(r_unit, v_sat, z_axis_body)

    # 3. Construct Rotation Matrix (Inertial to Body)
    # Columns of R are the body axes expressed in the inertial frame.
    R = np.array([x_axis_body, y_axis_body, z_axis_body]).T

    # 4. Convert Rotation Matrix to Quaternion
    q = _rotation_matrix_to_quaternion(R)
        
    return q

def generate_quaternions_over_period(
    satellite: EarthSatellite,
    ts: Timescale,
    start_time: datetime,
    end_time: datetime,
    interval: int
) -> List[Tuple[int, float, np.ndarray]]:
    """
    Generate nadir-pointing quaternion data for a given satellite over a specified time period.

    Args:
        satellite: A Skyfield `EarthSatellite` object.
        ts: A Skyfield `Timescale` object.
        start_time: The start datetime for quaternion generation.
        end_time: The end datetime for quaternion generation.
        interval: The time interval in seconds between quaternion calculations.

    Returns:
        A list of tuples, where each tuple contains:
        (MJD_days, MJD_seconds_of_day, quaternion_array [qw, qx, qy, qz]).
        Returns an empty list if no quaternions are generated (e.g., start_time > end_time).
    """
    quaternions: List[Tuple[int, float, np.ndarray]] = []
    current_time = start_time

    while current_time <= end_time:
        geocentric = satellite.at(ts.utc(current_time.year, current_time.month, current_time.day, 
                                        current_time.hour, current_time.minute, current_time.second))
        r_sat = geocentric.position.km
        v_sat = geocentric.velocity.km_per_s

        jd = ts.utc(current_time.year, current_time.month, current_time.day, 
                    current_time.hour, current_time.minute, current_time.second).tt
        days, seconds = _julian_date_to_mjd(jd)

        quaternion = _calculate_quaternion(r_sat, v_sat)
        quaternions.append((days, seconds, quaternion))

        current_time += timedelta(seconds=interval)

    return quaternions

def parse_ccsds_oem_file(file_path: str) -> Tuple[Dict[str, str], List[Tuple[Tuple[int, float], np.ndarray, np.ndarray]]]:
    """
    Parse a CCSDS OEM (Orbit Ephemeris Message) file to extract metadata and state vectors.

    Assumes the OEM file provides position in km and velocity in km/s in the EME2000 frame.
    Time is expected as MJD day and MJD seconds of the day.

    Args:
        file_path: The path to the CCSDS OEM file.

    Returns:
        A tuple containing:
        - metadata (Dict[str, str]): A dictionary of metadata like OBJECT_NAME, OBJECT_ID.
        - state_vectors (List[Tuple[Tuple[int, float], np.ndarray, np.ndarray]]):
          A list of state vectors. Each state vector is a tuple:
          ((MJD_day, MJD_seconds_of_day), position_vector_km, velocity_vector_km_s).

    Raises:
        Exception: If the file is not found, cannot be parsed, META_STOP is missing,
                   or no state vectors are found after META_STOP.
    """
    metadata: Dict[str, str] = {"OBJECT_NAME": "UNKNOWN", "OBJECT_ID": "UNKNOWN"}
    state_vectors: List[Tuple[Tuple[int, float], np.ndarray, np.ndarray]] = []
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
                        logging.warning(f"Skipping malformed data line in OEM file: {line} - {e}")
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

def generate_quaternions_from_oem_vectors(
    oem_state_vectors: List[Tuple[Tuple[int, float], np.ndarray, np.ndarray]]
) -> List[Tuple[int, float, np.ndarray]]:
    """
    Generate nadir-pointing quaternion data from a list of parsed OEM state vectors.

    Args:
        oem_state_vectors: A list of state vectors, where each state vector is a tuple:
                           ((MJD_day, MJD_seconds_of_day), position_vector_km, velocity_vector_km_s).

    Returns:
        A list of tuples, where each tuple contains:
        (MJD_days, MJD_seconds_of_day, quaternion_array [qw, qx, qy, qz]).
        Returns an empty list if the input `oem_state_vectors` is empty.
    """
    quaternions_data: List[Tuple[int, float, np.ndarray]] = []
    if not oem_state_vectors:
        return quaternions_data

    for (mjd_day, mjd_seconds_of_day), r_sat, v_sat in oem_state_vectors:
        # `calculate_quaternion` expects r_sat, v_sat in km and km/s, which OEM provides.
        quaternion = _calculate_quaternion(r_sat, v_sat)
        # The output format requires MJD day and MJD seconds of day.
        quaternions_data.append((mjd_day, mjd_seconds_of_day, quaternion))
    
    return quaternions_data

def write_aem_file(
    output_path: str,
    data_lines: List[Tuple[int, float, np.ndarray]],
    object_name: str,
    object_id: str,
    originator: str,
    generator_comment: str
) -> None:
    """
    Writes quaternion data to a CCSDS Attitude Ephemeris Message (AEM) file.

    Args:
        output_path: The full path to the output AEM file.
        data_lines: A list of tuples, where each tuple is 
                    (MJD_days, MJD_seconds_of_day, quaternion_array [qw, qx, qy, qz]).
        object_name: The name of the object (satellite).
        object_id: The ID of the object (satellite).
        originator: The originator of the AEM file (e.g., "VTS").
        generator_comment: A comment string describing the generator.

    Raises:
        IOError: If there's an issue writing to the file.
    """
    logging.info(f"Writing AEM file to: {output_path}")
    try:
        with open(output_path, 'w') as f:
            f.write(f"""CIC_AEM_VERS = 1.0
COMMENT {generator_comment}
CREATION_DATE = {(datetime.now(timezone.utc)).isoformat(timespec='microseconds')}
ORIGINATOR = {originator}

META_START

OBJECT_NAME = {object_name}
OBJECT_ID = {object_id}
REF_FRAME_A = EME2000
REF_FRAME_B = UNDEFINED
ATTITUDE_DIR = A2B
TIME_SYSTEM = UTC
ATTITUDE_TYPE = QUATERNION

META_STOP
\n
""")
            for days, seconds, quaternion_data_point in data_lines:
                f.write(_generate_ccsds_quaternion_data(days, seconds, quaternion_data_point) + '\n')
        logging.info(f"Successfully wrote AEM file: {output_path}")
    except IOError as e:
        logging.error(f"IOError writing AEM file {output_path}: {e}", exc_info=True)
        raise # Re-raise the exception to be caught by the GUI
    except Exception as e: # Catch any other unexpected errors during file writing
        logging.error(f"Unexpected error writing AEM file {output_path}: {e}", exc_info=True)
        raise