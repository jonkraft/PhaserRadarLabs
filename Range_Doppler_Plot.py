# %%
# Copyright (C) 2019 Analog Devices, Inc.
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#     - Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     - Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in
#       the documentation and/or other materials provided with the
#       distribution.
#     - Neither the name of Analog Devices, Inc. nor the names of its
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#     - The use of this software may or may not infringe the patent rights
#       of one or more patent holders.  This license does not release you
#       from the requirement that you obtain separate licenses from these
#       patent holders to use this software.
#     - Use of the software either in source or binary form, must be run
#       on or directly connected to an Analog Devices Inc. component.
#
# THIS SOFTWARE IS PROVIDED BY ANALOG DEVICES "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED.
#
# IN NO EVENT SHALL ANALOG DEVICES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, INTELLECTUAL PROPERTY
# RIGHTS, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
# THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''FMCW Range Doppler Demo with Phaser (CN0566)
   Jon Kraft, Jan 20 2024'''

# Imports
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time

'''This script uses the new Pluto TDD engine
   As of Jan 2024, this is in the "dev_phaser_merge" branch of https://github.com/analogdevicesinc/pyadi-iio
   Also, make sure your Pluto firmware is updated to rev 0.38 (or later)
'''
#sys.path.insert(0,'/home/analog/cn0566_merge/pyadi-iio/')
import adi
print(adi.__version__)

# Parameters
sample_rate = 5e6 
center_freq = 2.1e9
signal_freq = int(sample_rate/10)
ramp_time = 200  # us
num_chirps = 128

# Instantiate all the Devices
rpi_ip = "ip:phaser.local"  # IP address of the Raspberry Pi
sdr_ip = "ip:192.168.2.1"  # "192.168.2.1, or pluto.local"  # IP address of the Transceiver Block
my_sdr = adi.ad9361(uri=sdr_ip)
my_phaser = adi.CN0566(uri=rpi_ip, sdr=my_sdr)

# Initialize both ADAR1000s, set gains to max, and all phases to 0
my_phaser.configure(device_mode="rx")
my_phaser.load_gain_cal()
my_phaser.load_phase_cal()
for i in range(0, 8):
    my_phaser.set_chan_phase(i, 0)

gain_list = [8, 34, 84, 127, 127, 84, 34, 8]  # Blackman taper
for i in range(0, len(gain_list)):
    my_phaser.set_chan_gain(i, gain_list[i], apply_cal=True)

# Setup Raspberry Pi GPIO states
try:
    my_phaser._gpios.gpio_tx_sw = 0  # 0 = TX_OUT_2, 1 = TX_OUT_1
    my_phaser._gpios.gpio_vctrl_1 = 1 # 1=Use onboard PLL/LO source  (0=disable PLL and VCO, and set switch to use external LO input)
    my_phaser._gpios.gpio_vctrl_2 = 1 # 1=Send LO to transmit circuitry  (0=disable Tx path, and send LO to LO_OUT)
except:
    my_phaser.gpios.gpio_tx_sw = 0  # 0 = TX_OUT_2, 1 = TX_OUT_1
    my_phaser.gpios.gpio_vctrl_1 = 1 # 1=Use onboard PLL/LO source  (0=disable PLL and VCO, and set switch to use external LO input)
    my_phaser.gpios.gpio_vctrl_2 = 1 # 1=Send LO to transmit circuitry  (0=disable Tx path, and send LO to LO_OUT)

# Configure SDR Rx
my_sdr.sample_rate = int(sample_rate)
my_sdr.rx_lo = int(center_freq)   # set this to output_freq - (the freq of the HB100)
my_sdr.rx_enabled_channels = [0, 1]   # enable Rx1 (voltage0) and Rx2 (voltage1)
my_sdr.gain_control_mode_chan0 = 'manual'  # manual or slow_attack
my_sdr.gain_control_mode_chan1 = 'manual'  # manual or slow_attack
my_sdr.rx_hardwaregain_chan0 = int(30)   # must be between -3 and 70
my_sdr.rx_hardwaregain_chan1 = int(30)   # must be between -3 and 70
# Configure SDR Tx
my_sdr.tx_lo = int(center_freq)
my_sdr.tx_enabled_channels = [0, 1]
my_sdr.tx_cyclic_buffer = True      # must set cyclic buffer to true for the tdd burst mode.  Otherwise Tx will turn on and off randomly
my_sdr.tx_hardwaregain_chan0 = -88   # must be between 0 and -88
my_sdr.tx_hardwaregain_chan1 = -0   # must be between 0 and -88

# Read properties
print("RX LO %s" % (my_sdr.rx_lo))

# Configure the ADF4159 Rampling PLL
output_freq = 12.145e9
BW = 500e6
num_steps = ramp_time  # in general it works best if there is 1 step per us
my_phaser.frequency = int(output_freq / 4)  # Output frequency divided by 4
my_phaser.freq_dev_range = int(
    BW / 4
)  # frequency deviation range in Hz.  This is the total freq deviation of the complete freq ramp
my_phaser.freq_dev_step = int(
    (BW/4) / num_steps
)  # frequency deviation step in Hz.  This is fDEV, in Hz.  Can be positive or negative
my_phaser.freq_dev_time = int(
    ramp_time
)  # total time (in us) of the complete frequency ramp
print("requested freq dev time (us) = ", ramp_time)
ramp_time = my_phaser.freq_dev_time
print("actual freq dev time (us) = ", ramp_time)
my_phaser.delay_word = 4095  # 12 bit delay word.  4095*PFD = 40.95 us.  For sawtooth ramps, this is also the length of the Ramp_complete signal
my_phaser.delay_clk = "PFD"  # can be 'PFD' or 'PFD*CLK1'
my_phaser.delay_start_en = 0  # delay start
my_phaser.ramp_delay_en = 0  # delay between ramps.
my_phaser.trig_delay_en = 0  # triangle delay
my_phaser.ramp_mode = "single_sawtooth_burst"  # ramp_mode can be:  "disabled", "continuous_sawtooth", "continuous_triangular", "single_sawtooth_burst", "single_ramp_burst"
my_phaser.sing_ful_tri = (
    0  # full triangle enable/disable -- this is used with the single_ramp_burst mode
)
my_phaser.tx_trig_en = 1  # start a ramp with TXdata
my_phaser.enable = 0  # 0 = PLL enable.  Write this last to update all the registers

# %%
# Configure TDD controller
sdr_pins = adi.one_bit_adc_dac(sdr_ip)
sdr_pins.gpio_tdd_ext_sync = True # If set to True, this enables external capture triggering using the L24N GPIO on the Pluto.  When set to false, an internal trigger pulse will be generated every second
tdd = adi.tddn(sdr_ip)
sdr_pins.gpio_phaser_enable = True
tdd.enable = False         # disable TDD to configure the registers
tdd.sync_external = True
tdd.startup_delay_ms = 1
tdd.frame_length_ms = ramp_time/1e3 + 0.2    # each GPIO toggle is spaced this far apart
tdd.burst_count = num_chirps       # number of chirps in one continuous receive buffer

tdd.out_channel0_enable = True
tdd.out_channel0_polarity = False
tdd.out_channel0_on_ms = 0.01    # each GPIO pulse will be 100us (0.6ms - 0.5ms).  And the first trigger will happen 0.5ms into the buffer
tdd.out_channel0_off_ms = 0.2
tdd.out_channel1_enable = True
tdd.out_channel1_polarity = False
tdd.out_channel1_on_ms = 0
tdd.out_channel1_off_ms = 0.1
tdd.out_channel2_enable = False
tdd.enable = True

# buffer size needs to be greater than the frame_time
frame_time = tdd.frame_length_ms*tdd.burst_count   # time in ms
print("frame_time:  ", frame_time, "ms")
buffer_time = 0
power=12
while frame_time > buffer_time:     
    power=power+1
    buffer_size = int(2**power) 
    buffer_time = buffer_size/my_sdr.sample_rate*1000   # buffer time in ms
    if power==23:
        break     # max pluto buffer size is 2**23, but for tdd burst mode, set to 2**22
print("buffer_size:", buffer_size)
my_sdr.rx_buffer_size = buffer_size
print("buffer_time:", buffer_time, " ms")  

# Create a sinewave waveform
#fs = int(my_sdr.sample_rate)
fs = sample_rate
print("sample_rate:", fs)
N = buffer_size
fc = int(signal_freq / (fs / N)) * (fs / N)
ts = 1 / float(fs)
t = np.arange(0, N * ts, ts)
i = np.cos(2 * np.pi * t * fc) * 2 ** 14
q = np.sin(2 * np.pi * t * fc) * 2 ** 14
iq = 0.9* (i + 1j * q)

my_sdr._ctx.set_timeout(30000)
my_sdr._rx_init_channels() 

# Send data
my_sdr.tx([iq, iq])

# %%
PRI = tdd.frame_length_ms / 1e3
PRF = 1 / PRI
num_bursts = tdd.burst_count

# Split into frames
N_frame = int(PRI / ts)

# Obtain range-FFT x-axis
c = 3e8
wavelength = c / (output_freq - center_freq)
ramp_time_s = ramp_time / 1e6
slope = BW / ramp_time_s
freq = np.linspace(-fs / 2, fs / 2, N_frame)
dist = (freq - signal_freq) * c / (2 * slope)

# Resolutions
R_res = c / (2 * BW)
v_res = wavelength / (2 * num_bursts * PRI)

# Doppler spectrum limits
max_doppler_freq = PRF / 2
max_doppler_vel = max_doppler_freq * wavelength / 2

# First ramp starts with some offset (as defined in the TDD section above)
start_offset_time = tdd.out_channel0_on_ms/1e3

# From start of each ramp, how many "good" points do we want?
# For best freq linearity, stay away from the start of the ramps
begin_offset_time = 0.02e-3
good_ramp_time = ramp_time_s - begin_offset_time
good_ramp_samples = int(good_ramp_time * fs)
start_offset_samples = int((start_offset_time+begin_offset_time)*fs)


# %%
range_doppler_fig, ax = plt.subplots(figsize=(14, 7))

extent = [-max_doppler_vel, max_doppler_vel, dist.min(), dist.max()]

# %%    
# Collect data
my_phaser.gpios.gpio_burst = 0
my_phaser.gpios.gpio_burst = 1
my_phaser.gpios.gpio_burst = 0
data = my_sdr.rx()
chan1 = data[0]
chan2 = data[1]
sum_data = chan1+chan2

# Process data
# Make a 2D array of the chirps for each burst
rx_bursts = np.zeros((num_bursts, good_ramp_samples), dtype=complex)
for burst in range(num_bursts):
    start_index = start_offset_samples + (burst) * N_frame
    stop_index = start_index + good_ramp_samples
    rx_bursts[burst] = sum_data[start_index:stop_index]

rx_bursts_fft = np.fft.fftshift(abs(np.fft.fft2(rx_bursts)))


# %%    
i = 0
cmn = ''
def get_radar_data():
    global range_doppler
    # Collect data
    my_phaser.gpios.gpio_burst = 0
    my_phaser.gpios.gpio_burst = 1
    my_phaser.gpios.gpio_burst = 0
    data = my_sdr.rx()
    chan1 = data[0]
    chan2 = data[1]
    sum_data = chan1+chan2

    # Process data
    # Make a 2D array of the chirps for each burst
    rx_bursts = np.zeros((num_bursts, good_ramp_samples), dtype=complex)
    for burst in range(num_bursts):
        start_index = start_offset_samples + (burst) * N_frame
        stop_index = start_index + good_ramp_samples
        rx_bursts[burst] = sum_data[start_index:stop_index]
    
    rx_bursts_fft = np.fft.fftshift(abs(np.fft.fft2(rx_bursts)))
    range_doppler_data = np.log10(rx_bursts_fft).T
    plot_data = range_doppler_data
    #plot_data = np.clip(plot_data, 0, 6)  # clip the data to control the max spectrogram scale
    return plot_data
# %%
    
plot_data = np.log10(rx_bursts_fft).T
#plot_data = np.clip(plot_data, 0, 6)  # clip the data to control the max spectrogram scale
    
cmaps = ['inferno', 'plasma']
cmn = cmaps[0]
try:
    range_doppler = ax.imshow(plot_data, aspect='auto',
        extent=extent, origin='lower', cmap=matplotlib.colormaps.get_cmap(cmn),
        )
except:
    print("Using an older version of MatPlotLIB")
    from matplotlib.cm import get_cmap
    range_doppler = ax.imshow(plot_data, aspect='auto', vmin=0, vmax=8,
        extent=extent, origin='lower', cmap=get_cmap(cmn),
        )
ax.set_title('Range Doppler Spectrum', fontsize=24)
ax.set_xlabel('Velocity [m/s]', fontsize=22)
ax.set_ylabel('Range [m]', fontsize=22)

max_range = 10
ax.set_xlim([-6, 6])
ax.set_ylim([0, max_range])
ax.set_yticks(np.arange(2, max_range, 2))
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

print("sample_rate = ", sample_rate/1e6, "MHz, ramp_time = ", ramp_time, "us, num_chirps = ", num_chirps)
print("CTRL + c to stop the loop")
try:
    while True:
        plot_data = get_radar_data()
        range_doppler.set_data(plot_data)
        plt.show(block=False)
        plt.pause(.1)
except KeyboardInterrupt:  # press ctrl-c to stop the loop
    pass

# %%
# Pluto transmit shutdown
my_sdr.tx_destroy_buffer()
print("Buffer Destroyed!")

# # To disable TDD and revert to non-TDD (standard) mode
tdd.enable = False
sdr_pins.gpio_phaser_enable = False
tdd.out_channel1_polarity = not(sdr_pins.gpio_phaser_enable)
tdd.out_channel2_polarity = sdr_pins.gpio_phaser_enable
tdd.enable = True
tdd.enable = False
