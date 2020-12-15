from pykalman import KalmanFilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
# importing GPS data 
df = pd.read_csv('C:/Users/mun/Desktop/Research/Ny mappe/GPS_track.csv')
df.head(1000)
# data preparation for 2-D Kalman filter which require position (latitude, longitude)
# and time. 
lat = np.array([df.latitude])
print(lat)
long = np.array([df.longitude])
print(long)
print(len(long[0]))
for i in range(len(long)):
    print(long[i][0])
for i in range(len(lat[0])):
    print(lat[0][i])
print(len(lat[0]))
print(len(long[0]))

#length of the arrays. the arrays should always have the same length
lng=len(lat[0])
print(lng)
for index in range(lng):
    print(lat[0][index])
    print(long[0][index])
for index in range (lng):
    np.array((lat[0][index], long[0][index]))
coord1 = [list(i) for i in zip (lat[0],long[0])]
print(coord1)
coord = list(zip(lat[0],long[0]))
print(coord)  

#----Kalamn Filter-----Functionality 
measurements = np.asarray(coord1)  # coord1 is our prepared data file

initial_state_mean = [measurements[0, 0],
                      0,
                      measurements[0, 1],
                      0]

transition_matrix = [[1, 1, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 1],
                     [0, 0, 0, 1]]

observation_matrix = [[1, 0, 0, 0],
                      [0, 0, 1, 0]]

kf1 = KalmanFilter(transition_matrices = transition_matrix,
                  observation_matrices = observation_matrix,
                  initial_state_mean = initial_state_mean)
kf1 = kf1.em(measurements, n_iter=5)
time_before = time.time()
n_real_time = 3

kf2 = KalmanFilter(transition_matrices = transition_matrix,
                  observation_matrices = observation_matrix,
                  initial_state_mean = initial_state_mean,
                  observation_covariance = 10*kf1.observation_covariance,
                  em_vars=['transition_covariance', 'initial_state_covariance'])

kf2 = kf2.em(measurements[:-n_real_time, :], n_iter=5)
(filtered_state_means, filtered_state_covariances) = kf2.filter(measurements[:-n_real_time,:])

print("Time to build and train kf2: %s seconds" % (time.time() - time_before))

x_now = filtered_state_means[1, :]
P_now = filtered_state_covariances[1, :]
x_new = np.zeros((n_real_time, filtered_state_means.shape[1]))
i = 0

for measurement in measurements[-n_real_time:, :]:
    time_before = time.time()
    (x_now, P_now) = kf2.filter_update(filtered_state_mean = x_now,
                                       filtered_state_covariance = P_now,
                                       observation = measurement)
    print("Time to update kf2: %s seconds" % (time.time() - time_before))
    x_new[i, :] = x_now
    i = i + 1

plt.figure(1)
old_times = range(measurements.shape[0] - n_real_time)
new_times = range(measurements.shape[0]-n_real_time, measurements.shape[0])
plt.plot(old_times, filtered_state_means[:, 0], 'b--',
         old_times, filtered_state_means[:, 2], 'r--',
         new_times, x_new[:, 0], 'b-',
         new_times, x_new[:, 2], 'r-')

plt.show()
latitude = np.array(filtered_state_means[:, 0])
longitude = np.array(filtered_state_means[:, 2])

df1 = pd.DataFrame({'latitude': latitude, 'longitude': list(longitude)}, columns=['latitude', 'longitude'])
print(df1)
df1.to_csv('C:/Users/mun/Desktop/Research/Ny mappe/Filtered_new1_data.csv')

#------ Kalman filter smoothing process------
(smoothed_state_means, smoothed_state_covariances) = kf2.smooth(measurements)
kf3 = KalmanFilter(transition_matrices = transition_matrix,
                  observation_matrices = observation_matrix,
                  initial_state_mean = initial_state_mean,
                  observation_covariance = 10*kf2.observation_covariance,
                  em_vars=['transition_covariance', 'initial_state_covariance'])

kf3 = kf3.em(measurements, n_iter=5)
(smoothed_state_means, smoothed_state_covariances)  = kf3.smooth(measurements)

plt.figure(2)
times = range(measurements.shape[0])
plt.plot(times, measurements[:, 0], 'bo',
         times, measurements[:, 1], 'ro',
         times, smoothed_state_means[:, 0], 'b',
         times, smoothed_state_means[:, 2], 'r',)
plt.show()
plt.figure(3)
times = range(measurements.shape[0])
plt.plot(times, measurements[:, 0], 'bo',
         times, measurements[:, 1], 'ro',
         times, smoothed_state_means[:, 0], 'b--',
         times, smoothed_state_means[:, 1], 'r--',)
plt.figure(4)
latitude = np.array(smoothed_state_means[:, 0])
longitude = np.array(smoothed_state_means[:, 2])
df2 = pd.DataFrame({'latitude': latitude, 'longitude': list(longitude)}, columns=['latitude', 'longitude'])
times = range(measurements.shape[0])
plt.plot(times, measurements[:, 0], 'bo',
         times, smoothed_state_means[:, 0], 'b')
plt.axis([0, 920, 54.7, 55])
plt.show()
times = range(measurements.shape[0])
plt.plot(times, measurements[:, 1], 'ro',
         times, smoothed_state_means[:, 2], 'r--')
plt.axis([0, 920, 11.7, 12])
plt.show()
df2.to_csv('C:/Users/mun/Desktop/Research/Ny mappe/Filtered_smooth_data1.csv')


# End....
