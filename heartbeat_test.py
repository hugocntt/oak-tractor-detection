from pymavlink import mavutil
 
connection = mavutil.mavlink_connection("/dev/ttyAMA10", baud=921600)
print("waiting for heartbeat...")
connection.wait_heartbeat()
print("connected! system:", connection.target_system)
