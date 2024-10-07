# %%
# Copyright (C) 2024 Analog Devices, Inc.
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
   Updated for new TDD engine (rev 0.39 Pluto firmware)
   Added pulse canceller MTI filter
   Jon Kraft, Sept 25 2024'''

# %%
# Imports
import sys
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
plt.close('all')

'''This script uses the new Pluto TDD engine
   As of March 2024, this is in the main branch of https://github.com/analogdevicesinc/pyadi-iio
   Also, make sure your Pluto firmware is updated to rev 0.39 (or later)
'''
import adi
print(adi.__version__)

'''Key Parameters'''
sample_rate = 4e6 
center_freq = 2.1e9
signal_freq = 100e3
rx_gain = 60   # must be between -3 and 70
tx_gain = 0   # must be between 0 and -88
output_freq = 9.9e9
chirp_BW = 500e6
ramp_time = 300  # us
num_chirps = 256
max_range = 100
min_scale = 0
max_scale = 100
plot_data = True
mti_filter = True
save_data = False   # saves data for later processing (use "Range_Doppler_Processing.py")
f = "saved_radar_data.npy"

# %%
""" Program the basic hardware settings
"""
# Instantiate all the Devices
rpi_ip = "ip:phaser.local"  # IP address of the Raspberry Pi
sdr_ip = "ip:192.168.2.1"  # "192.168.2.1, or pluto.local"  # IP address of the Transceiver Block
#sdr_ip = "ip:phaser.local:50901"  # using IIO context port forwarding
my_sdr = adi.ad9361(uri=sdr_ip)
my_phaser = adi.CN0566(uri=rpi_ip, sdr=my_sdr)

# Initialize both ADAR1000s, set gains to max, and all phases to 0
my_phaser.configure(device_mode="rx")
my_phaser.element_spacing = 0.014
my_phaser.load_gain_cal()
my_phaser.load_phase_cal()
for i in range(0, 8):
    my_phaser.set_chan_phase(i, 0)

gain_list = [127] * 8
#gain_list = [8, 34, 84, 127, 127, 84, 34, 8]  # Blackman taper
for i in range(0, len(gain_list)):
    my_phaser.set_chan_gain(i, gain_list[i], apply_cal=True)

# Setup Raspberry Pi GPIO states
my_phaser._gpios.gpio_tx_sw = 0  # 0 = TX_OUT_2, 1 = TX_OUT_1
my_phaser._gpios.gpio_vctrl_1 = 1 # 1=Use onboard PLL/LO source  (0=disable PLL and VCO, and set switch to use external LO input)
my_phaser._gpios.gpio_vctrl_2 = 1 # 1=Send LO to transmit circuitry  (0=disable Tx path, and send LO to LO_OUT)

# Configure SDR Rx
my_sdr.sample_rate = int(sample_rate)
my_sdr.rx_lo = int(center_freq)
my_sdr.rx_enabled_channels = [0, 1]   # enable Rx1 and Rx2
my_sdr.gain_control_mode_chan0 = 'manual'  # manual or slow_attack
my_sdr.gain_control_mode_chan1 = 'manual'  # manual or slow_attack
my_sdr.rx_hardwaregain_chan0 = int(rx_gain)   # must be between -3 and 70
my_sdr.rx_hardwaregain_chan1 = int(rx_gain)   # must be between -3 and 70

# Configure SDR Tx
my_sdr.tx_lo = int(center_freq)
my_sdr.tx_enabled_channels = [0, 1]
my_sdr.tx_cyclic_buffer = True      # must set cyclic buffer to true for the tdd burst mode
my_sdr.tx_hardwaregain_chan0 = -88   # must be between 0 and -88
my_sdr.tx_hardwaregain_chan1 = int(tx_gain)   # must be between 0 and -88

# Configure the ADF4159 Ramping PLL
vco_freq = int(output_freq + signal_freq + center_freq)
BW = chirp_BW
num_steps = int(ramp_time)    # in general it works best if there is 1 step per us
my_phaser.frequency = int(vco_freq / 4)
my_phaser.freq_dev_range = int(BW / 4)      # total freq deviation of the complete freq ramp in Hz
my_phaser.freq_dev_step = int((BW / 4) / num_steps)  # This is fDEV, in Hz.  Can be positive or negative
my_phaser.freq_dev_time = int(ramp_time)  # total time (in us) of the complete frequency ramp
print("requested freq dev time (us) = ", ramp_time)
my_phaser.delay_word = 4095  # 12 bit delay word.  4095*PFD = 40.95 us.  For sawtooth ramps, this is also the length of the Ramp_complete signal
my_phaser.delay_clk = "PFD"  # can be 'PFD' or 'PFD*CLK1'
my_phaser.delay_start_en = 0  # delay start
my_phaser.ramp_delay_en = 0  # delay between ramps.
my_phaser.trig_delay_en = 0  # triangle delay
my_phaser.ramp_mode = "single_sawtooth_burst"  # ramp_mode can be:  "disabled", "continuous_sawtooth", "continuous_triangular", "single_sawtooth_burst", "single_ramp_burst"
my_phaser.sing_ful_tri = 0  # full triangle enable/disable -- this is used with the single_ramp_burst mode
my_phaser.tx_trig_en = 1  # start a ramp with TXdata
my_phaser.enable = 0  # 0 = PLL enable.  Write this last to update all the registers

# %%
""" Synchronize chirps to the start of each Pluto receive buffer
"""
# Configure TDD controller
sdr_pins = adi.one_bit_adc_dac(sdr_ip)
sdr_pins.gpio_tdd_ext_sync = True # If set to True, this enables external capture triggering using the L24N GPIO on the Pluto.  When set to false, an internal trigger pulse will be generated every second
tdd = adi.tddn(sdr_ip)
sdr_pins.gpio_phaser_enable = True
tdd.enable = False         # disable TDD to configure the registers
tdd.sync_external = True
tdd.startup_delay_ms = 0
PRI_ms = ramp_time/1e3 + 0.2
tdd.frame_length_ms = PRI_ms    # each chirp is spaced this far apart
#tdd.frame_length_raw = PRI_ms/1000 * 2 * sample_rate
tdd.burst_count = num_chirps       # number of chirps in one continuous receive buffer

tdd.channel[0].enable = True
tdd.channel[0].polarity = False
tdd.channel[0].on_raw = 0
tdd.channel[0].off_raw = 10
tdd.channel[1].enable = True
tdd.channel[1].polarity = False
tdd.channel[1].on_raw = 0
tdd.channel[1].off_raw = 10
tdd.channel[2].enable = True
tdd.channel[2].polarity = False
tdd.channel[2].on_raw = 0
tdd.channel[2].off_raw = 10
tdd.enable = True

# From start of each ramp, how many "good" points do we want?
# For best freq linearity, stay away from the start of the ramps
ramp_time = int(my_phaser.freq_dev_time) # - begin_offset_time)
ramp_time_s = ramp_time / 1e6
begin_offset_time = 0.1 * ramp_time_s   # time in seconds
print("actual freq dev time = ", ramp_time)
good_ramp_samples = int((ramp_time_s - begin_offset_time) * sample_rate)
start_offset_time = tdd.channel[0].on_ms/1e3 + begin_offset_time
start_offset_samples = int(start_offset_time * sample_rate)

# size the fft for the number of ramp data points
power=8
fft_size = int(2**power)
num_samples_frame = int(tdd.frame_length_ms/1000*sample_rate)
while num_samples_frame > fft_size:     
    power=power+1
    fft_size = int(2**power) 
    if power==18:
        break
print("fft_size =", fft_size)

# Pluto receive buffer size needs to be greater than total time for all chirps
total_time = tdd.frame_length_ms * num_chirps   # time in ms
print("Total Time for all Chirps:  ", total_time, "ms")
buffer_time = 0
power=12
while total_time > buffer_time:     
    power=power+1
    buffer_size = int(2**power) 
    buffer_time = buffer_size/sample_rate*1000   # buffer time in ms
    if power==23:
        break     # max pluto buffer size is 2**23, but for tdd burst mode, set to 2**22
print("buffer_size:", buffer_size)
my_sdr.rx_buffer_size = buffer_size
print("buffer_time:", buffer_time, " ms")

# %%
""" Calculate ramp parameters
"""
PRI_s = PRI_ms / 1e3
PRF = 1 / PRI_s
num_bursts = tdd.burst_count

# Split into frames
N_frame = int(PRI_s * float(sample_rate))

# Obtain range-FFT x-axis
c = 3e8
wavelength = c / output_freq
slope = BW / ramp_time_s
freq = np.linspace(-sample_rate / 2, sample_rate / 2, N_frame)
dist = (freq - signal_freq) * c / (2 * slope)

# Resolutions
R_res = c / (2 * BW)
v_res = wavelength / (2 * num_bursts * PRI_s)

# Doppler spectrum limits
max_doppler_freq = PRF / 2
max_doppler_vel = max_doppler_freq * wavelength / 2


# %%
""" Create a sinewave waveform for Pluto's transmitter
"""
# Create a sinewave waveform
N = int(2**18)
fc = int(signal_freq)
ts = 1 / float(sample_rate)
t = np.arange(0, N * ts, ts)
i = np.cos(2 * np.pi * t * fc) * 2 ** 14
q = np.sin(2 * np.pi * t * fc) * 2 ** 14
iq = 0.9* (i + 1j * q)

# transmit data from Pluto
my_sdr.tx([iq, iq])


# %%
# Function to collect data
i = 0
cmn = ''
def get_radar_data():
    global range_doppler
    # Collect data
    my_phaser._gpios.gpio_burst = 0
    my_phaser._gpios.gpio_burst = 1
    my_phaser._gpios.gpio_burst = 0
    data = my_sdr.rx()
    chan1 = data[0]
    chan2 = data[1]
    sum_data = chan1+chan2

    # Process data
    # Make a 2D array of the chirps for each burst
    rx_bursts = np.zeros((num_bursts, good_ramp_samples), dtype=complex)
    for burst in range(num_bursts):
        start_index = start_offset_samples + burst * N_frame
        stop_index = start_index + good_ramp_samples
        rx_bursts[burst] = sum_data[start_index:stop_index]
    
    return rx_bursts

def freq_process(data):
    rx_bursts_fft = np.fft.fftshift(abs(np.fft.fft2(data)))
    range_doppler_data = np.log10(rx_bursts_fft).T
    range_doppler_data = np.clip(range_doppler_data, min_scale, max_scale)  # clip the data to control the max spectrogram scale
    return range_doppler_data


# %%
    
rx_bursts = get_radar_data()
radar_data = freq_process(rx_bursts)
all_data = []

if plot_data == True:
    range_doppler_fig, ax = plt.subplots(figsize=(14, 7))
    extent = [-max_doppler_vel, max_doppler_vel, dist.min(), dist.max()]
    cmaps = ['inferno', 'plasma']
    cmn = cmaps[0]
    try:
        range_doppler = ax.imshow(radar_data, aspect='auto',
            extent=extent, origin='lower', cmap=matplotlib.colormaps.get_cmap(cmn),
            )
    except:
        print("Using an older version of MatPlotLIB")
        from matplotlib.cm import get_cmap
        range_doppler = ax.imshow(radar_data, aspect='auto', vmin=0, vmax=8,
            extent=extent, origin='lower', cmap=get_cmap(cmn),
            )
    ax.set_title('Range Doppler Spectrum', fontsize=24)
    ax.set_xlabel('Velocity [m/s]', fontsize=22)
    ax.set_ylabel('Range [m]', fontsize=22)
    
    ax.set_xlim([-10, 10])
    ax.set_ylim([0, max_range])
    ax.set_yticks(np.arange(0, max_range, max_range/20))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    print("sample_rate = ", sample_rate/1e6, "MHz, ramp_time = ", ramp_time, "us, num_chirps = ", num_chirps)
    print("CTRL + c to stop the loop")

try:
    while True:
        rx_bursts = get_radar_data()
        if save_data == True:
            all_data.append(rx_bursts)
            print("save")
        if plot_data == True:
            if mti_filter == True:
                rx_chirps = []
                rx_chirps = rx_bursts
                num_samples = len(rx_chirps[0])
                # create 2 pulse canceller MTI array
                Chirp2P = np.ones([num_chirps, num_samples]) * 1j
                for chirp in range(num_chirps-1):
                    chirpI = rx_chirps[chirp,:]
                    chirpI1 = rx_chirps[chirp+1,:]
                    chirp_correlation = np.correlate(chirpI, chirpI1, 'valid')
                    angle_diff = np.angle(chirp_correlation, deg=False)  # returns radians
                    Chirp2P[chirp:] = chirpI1 - chirpI * np.exp(-1j*angle_diff[0])
                rx_bursts = Chirp2P
                
            radar_data = freq_process(rx_bursts)
            range_doppler.set_data(radar_data)
            plt.show(block=False)
            plt.pause(.1)
except KeyboardInterrupt:  # press ctrl-c to stop the loop
    pass

# %%
# Pluto transmit shutdown
my_sdr.tx_destroy_buffer()
print("Pluto Buffer Cleared!")
if save_data == True:
    np.save(f, all_data)
    np.save(f[:-4]+"_config.npy", [sample_rate, signal_freq, output_freq, num_chirps, chirp_BW, ramp_time_s, tdd.frame_length_ms])

