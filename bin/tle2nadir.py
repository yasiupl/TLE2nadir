import numpy as np
from datetime import datetime, timedelta, timezone
from skyfield.api import EarthSatellite, load
import argparse

def normalize(vector):
    """Normalize a vector."""
    return vector / np.linalg.norm(vector)

def calculate_quaternion(r_sat):
    """Calculate the quaternion for a nadir-pointing satellite."""
    
    # Normalize the position vector
    r_unit = normalize(r_sat)

    # Define nadir pointing direction as -r_unit
    nadir_direction = -r_unit

    # Assume default "up" vector as z-axis (0, 0, 1) in EME2000 frame
    up_vector = np.array([0, 0, 1])

    # Compute an orthonormal basis for the attitude
    z_axis = nadir_direction
    y_axis = normalize(np.cross(up_vector, z_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))

    # Rotation matrix from EME2000 to satellite body frame
    R = np.vstack((x_axis, y_axis, z_axis)).T

    # Convert rotation matrix to quaternion
    q = np.zeros(4)
    q[0] = np.sqrt(1.0 + R[0, 0] + R[1, 1] + R[2, 2]) / 2  # Real part first
    q[1] = (R[2, 1] - R[1, 2]) / (4 * q[0])
    q[2] = (R[0, 2] - R[2, 0]) / (4 * q[0])
    q[3] = (R[1, 0] - R[0, 1]) / (4 * q[0])

    return q

def julian_date_to_MJD(jd):
    """Convert Julian Date to CCSDS, expressing as days and seconds."""
    MJD_epoch =  2400000.5   # CCSDS time starts at January 1, 1958, 00:00:00
    delta_days = jd - MJD_epoch
    days = int(delta_days)
    seconds = (delta_days - days) * 86400
    return days, seconds

def mjd_to_datetime(mjd):
    """
    Convert a Modified Julian Date (MJD) to a Python datetime object.
    MJD epoch starts at 1858-11-17 00:00:00 UTC.

    Args:
        mjd (float): Modified Julian Date to convert.

    Returns:
        datetime: Corresponding datetime object in UTC.
    """
    mjd_epoch = datetime(1858, 11, 17, 0, 0, 0)  # MJD epoch
    delta = timedelta(days=mjd)
    return mjd_epoch + delta

def generate_ccsds_quaternion_data(days, seconds, quaternion):
    """Generate a CCSDS-compatible data format for quaternion."""
    
    return f"{days}\t{seconds:.6f}\t{quaternion[0]:.6f}\t{quaternion[1]:.6f}\t{quaternion[2]:.6f}\t{quaternion[3]:.6f}"

def calculate_quaternion(r_sat, v_sat):
    """Calculate the quaternion for a satellite with X pointing towards velocity (motion) and Z pointing towards nadir."""
    
    # Normalize the position and velocity vectors
    r_unit = normalize(r_sat)
    v_unit = normalize(v_sat)

    # Nadir direction is simply the opposite of the position vector
    nadir_direction = -r_unit

    # Assume that the velocity direction will align with the X-axis
    x_axis = v_unit
    
    # Compute Y-axis as the cross product between the Z and X axes
    y_axis = normalize(np.cross(nadir_direction, x_axis))
    
    # Ensure the Z-axis remains the nadir pointing direction
    z_axis = nadir_direction

    # Rotation matrix from EME2000 to satellite body frame
    R = np.vstack((x_axis, y_axis, z_axis)).T

    # Convert rotation matrix to quaternion
    q = np.zeros(4)
    q[0] = np.sqrt(1.0 + R[0, 0] + R[1, 1] + R[2, 2]) / 2  # Real part first
    q[1] = (R[2, 1] - R[1, 2]) / (4 * q[0])
    q[2] = (R[0, 2] - R[2, 0]) / (4 * q[0])
    q[3] = (R[1, 0] - R[0, 1]) / (4 * q[0])

    return q

def generate_quaternions_over_time(satellite, ts, start_time, end_time, interval):
    """Generate quaternions for a satellite over a set period of time, now with X-axis towards velocity."""
    quaternions = []
    current_time = start_time

    while current_time <= end_time:
        # Get satellite position and velocity in EME2000 frame (in km and km/s respectively)
        geocentric = satellite.at(ts.utc(current_time.year, current_time.month, current_time.day, 
                                        current_time.hour, current_time.minute, current_time.second))
        r_sat = geocentric.position.km
        v_sat = geocentric.velocity.km_per_s

        # Convert current time to JD1950
        jd = ts.utc(current_time.year, current_time.month, current_time.day, 
                    current_time.hour, current_time.minute, current_time.second).tt
        days, seconds = julian_date_to_MJD(jd)

        # Calculate quaternion with X pointing in the direction of velocity
        quaternion = calculate_quaternion(r_sat, v_sat)

        # Append raw quaternion data with time info
        quaternions.append((days, seconds, quaternion))

        # Increment time by the interval
        current_time = current_time + timedelta(seconds=interval)

    return quaternions

def example():
    # NORAD ID for the satellite
    norad_id = 39634

    # Load TLE data
    url = 'https://celestrak.org/NORAD/elements/gp.php?CATNR=' + str(norad_id)
    tle_data = load.tle_file(url)

    if len(tle_data) == 0:
        raise RuntimeError(f"Could not download TLE data for NORAD ID {norad_id}.")

    satellite = tle_data[0]
    ts = load.timescale()
    # start_time = datetime.utcnow()
    start_time = datetime(2025, 1, 13, 0, 0)
    end_time = start_time + timedelta(days=7)  # Generate quaternions for 10 minutes
    interval = 60  # Interval of 60 seconds

    # Generate quaternions
    quaternions = generate_quaternions_over_time(satellite, ts, start_time, end_time, interval)

    # Print quaternions in CCSDS format
    for days, seconds, quaternion in quaternions:
        print(generate_ccsds_quaternion_data(days, seconds, quaternion))

def main():
    # Set up the command line parser
    parser = argparse.ArgumentParser(description='Satellite Data Generator for VTS.')

    # Define command line arguments
    parser.add_argument('--vtsgentype', required=True, choices=['GenPosition', 'GenQuaternion', 'GenEuler', 'GenAxis', 'GenAngle', 'GenDirection', 'GenAltAz'],
                        help='The requested generation type')
    parser.add_argument('--vtsdefaultfile', required=True, help='Default output file path')
    parser.add_argument('--vtsmjddates', required=True, nargs=2, type=float, metavar=('StartMJD', 'EndMJD'),
                        help='Start and End dates as MJD')

    # Parse the arguments
    args = parser.parse_args()

    # Convert MJD to datetime for start and end dates
    start_date = mjd_to_datetime(args.vtsmjddates[0])
    end_date = mjd_to_datetime(args.vtsmjddates[1])

    # Load TLE data based on the command
    norad_id = 39634  # Example: Choose a valid NORAD ID
    url = f'https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}'
    tle_data = load.tle_file(url)

    if len(tle_data) == 0:
        raise RuntimeError(f"Could not download TLE data for NORAD ID {norad_id}.")
    satellite = tle_data[0]
    ts = load.timescale()

    # Set the interval (e.g., 60 seconds)
    interval = 60

    # Generate quaternions if the generation type is GenQuaternion
    if args.vtsgentype == 'GenQuaternion':
        quaternions = generate_quaternions_over_time(satellite, ts, start_date, end_date, interval)

        # Output quaternions to file
        with open(args.vtsdefaultfile, 'w') as file:
            for r_sat, quaternion in quaternions:
                file.write(f"Position: {r_sat} Quaternion: {quaternion}\n")

        print(f"Quaternion data written to {args.vtsdefaultfile}")

    # Implement other generation types here (e.g., 'GenPosition', 'GenEuler', etc.)
    # Example: 
    # if args.vtsgentype == 'GenPosition':
    #    process_position_data()

if __name__ == '__main__':
    main()