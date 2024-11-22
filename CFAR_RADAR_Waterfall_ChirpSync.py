#!/usr/bin/env python3
#  Must use Python 3
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

'''CFAR Radar Demo with Phaser (CN0566)
       and implementing one chirp synchronized FFT data collection
   Jon Kraft, March 2 2024'''
# %%
# Imports
import sys
import time
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from pyqtgraph.Qt import QtCore, QtGui
from target_detection_dbfs import cfar

'''This script uses the new Pluto TDD engine
   As of March 2024, this is in the main branch of https://github.com/analogdevicesinc/pyadi-iio
   Also, make sure your Pluto firmware is updated to rev 0.39 (or later)
'''
import adi
print(adi.__version__)

'''Key Parameters'''
sample_rate = 2e6
center_freq = 2.1e9
signal_freq = 100e3
rx_gain = 20   # must be between -3 and 70
output_freq = 10e9
default_chirp_bw = 500e6
ramp_time = 500      # ramp time in us
num_slices = 100     # this sets how much time will be displayed on the waterfall plot
fft_size = 1024 * 8
plot_freq = 100e3    # x-axis freq range to plot

# %%
""" Program the basic hardware settings
"""
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
my_phaser._gpios.gpio_tx_sw = 0  # 0 = TX_OUT_2, 1 = TX_OUT_1
my_phaser._gpios.gpio_vctrl_1 = 1 # 1=Use onboard PLL/LO source  (0=disable PLL and VCO, and set switch to use external LO input)
my_phaser._gpios.gpio_vctrl_2 = 1 # 1=Send LO to transmit circuitry  (0=disable Tx path, and send LO to LO_OUT)

# Configure SDR Rx
my_sdr.sample_rate = int(sample_rate)
sample_rate = int(my_sdr.sample_rate)
my_sdr.rx_lo = int(center_freq)  # set this to output_freq - (the freq of the HB100)
my_sdr.rx_enabled_channels = [0, 1]  # enable Rx1 and Rx2
my_sdr.gain_control_mode_chan0 = "manual"  # manual or slow_attack
my_sdr.gain_control_mode_chan1 = "manual"  # manual or slow_attack
my_sdr.rx_hardwaregain_chan0 = int(rx_gain)  # must be between -3 and 70
my_sdr.rx_hardwaregain_chan1 = int(rx_gain)  # must be between -3 and 70

# Configure SDR Tx
my_sdr.tx_lo = int(center_freq)
my_sdr.tx_enabled_channels = [0, 1]
my_sdr.tx_cyclic_buffer = True  # must set cyclic buffer to true for the tdd burst mode.  Otherwise Tx will turn on and off randomly
my_sdr.tx_hardwaregain_chan0 = -88  # must be between 0 and -88
my_sdr.tx_hardwaregain_chan1 = -0  # must be between 0 and -88

# Configure the ADF4159 Rampling PLL
vco_freq = int(output_freq + signal_freq + center_freq)
BW = default_chirp_bw
num_steps = int(ramp_time)    # in general it works best if there is 1 step per us
my_phaser.frequency = int(vco_freq / 4)
my_phaser.freq_dev_range = int(BW / 4)      # total freq deviation of the complete freq ramp in Hz
my_phaser.freq_dev_step = int((BW / 4) / num_steps)  # This is fDEV, in Hz.  Can be positive or negative
my_phaser.freq_dev_time = int(ramp_time)  # total time (in us) of the complete frequency ramp
print("requested freq dev time = ", ramp_time)
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
PRI_ms = ramp_time/1e3 + 1.0
tdd.frame_length_ms = PRI_ms    # each chirp is spaced this far apart
num_chirps = 1
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
ramp_time = int(my_phaser.freq_dev_time)
ramp_time_s = ramp_time / 1e6
begin_offset_time = 0.10 * ramp_time_s   # time in seconds
print("actual freq dev time = ", ramp_time)
good_ramp_samples = int((ramp_time_s-begin_offset_time) * sample_rate)
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
    buffer_time = buffer_size/my_sdr.sample_rate*1000   # buffer time in ms
    if power==23:
        break     # max pluto buffer size is 2**23, but for tdd burst mode, set to 2**22
print("buffer_size:", buffer_size)
my_sdr.rx_buffer_size = buffer_size
print("buffer_time:", buffer_time, " ms")

# %%
""" Calculate and print summary of ramp parameters
"""
c = 3e8
wavelength = c / output_freq
freq = np.linspace(-sample_rate / 2, sample_rate / 2, int(fft_size))
slope = BW / ramp_time_s
dist = (freq - signal_freq) * c / (2 * slope)
plot_dist = False

print(
    """
CONFIG:
Sample rate: {sample_rate}MHz
Num samples: 2^{Nlog2}
Bandwidth: {BW}MHz
Ramp time: {ramp_time}ms
Output frequency: {output_freq}MHz
IF: {signal_freq}kHz
""".format(
        sample_rate=sample_rate / 1e6,
        Nlog2=int(np.log2(my_sdr.rx_buffer_size)),
        BW=BW / 1e6,
        ramp_time=ramp_time / 1e3,
        output_freq=output_freq / 1e6,
        signal_freq=signal_freq / 1e3,
    )
)
    
# %%
""" Create a sinewave waveform for Pluto's transmitter
"""
N = int(2**18)
fc = int(signal_freq)
ts = 1 / float(sample_rate)
t = np.arange(0, N * ts, ts)
i = np.cos(2 * np.pi * t * fc) * 2 ** 14
q = np.sin(2 * np.pi * t * fc) * 2 ** 14
iq = 1 * (i + 1j * q)

# transmit data from Pluto
my_sdr._ctx.set_timeout(30000)
my_sdr._rx_init_channels()
my_sdr.tx([iq, iq])


# %%
""" Create QT GUI Window, Buttons, and Plots
"""
plot_threshold = False
cfar_toggle = False
class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive FFT")
        self.setGeometry(0, 0, 400, 400)  # (x,y, width, height)
        #self.setFixedWidth(600)
        self.setWindowState(QtCore.Qt.WindowMaximized)
        self.num_rows = 12
        self.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, False) #remove the window's close button
        self.UiComponents()
        self.show()

    # method for components
    def UiComponents(self):
        widget = QWidget()

        global layout, signal_freq, plot_freq
        layout = QGridLayout()

        # Control Panel
        control_label = QLabel("PHASER CFAR Targeting")
        font = control_label.font()
        font.setPointSize(24)
        control_label.setFont(font)
        font.setPointSize(12)
        control_label.setAlignment(Qt.AlignHCenter)  # | Qt.AlignVCenter)
        layout.addWidget(control_label, 0, 0, 1, 2)

        # Check boxes
        self.thresh_check = QCheckBox("Plot CFAR Threshold")
        font = self.thresh_check.font()
        font.setPointSize(10)
        self.thresh_check.setFont(font)
        self.thresh_check.stateChanged.connect(self.change_thresh)
        layout.addWidget(self.thresh_check, 2, 0)
        
        self.cfar_check = QCheckBox("Apply CFAR Threshold")
        font = self.cfar_check.font()
        self.cfar_check.setFont(font)
        self.cfar_check.stateChanged.connect(self.change_cfar)
        layout.addWidget(self.cfar_check, 2, 1)

        # Chirp bandwidth slider
        self.bw_slider = QSlider(Qt.Horizontal)
        self.bw_slider.setMinimum(100)
        self.bw_slider.setMaximum(500)
        self.bw_slider.setValue(int(default_chirp_bw / 1e6))
        self.bw_slider.setTickInterval(50)
        self.bw_slider.setMaximumWidth(200)
        self.bw_slider.setTickPosition(QSlider.TicksBelow)
        self.bw_slider.valueChanged.connect(self.get_range_res)
        layout.addWidget(self.bw_slider, 4, 0)

        self.set_bw = QPushButton("Set Chirp Bandwidth")
        self.set_bw.setMaximumWidth(200)
        self.set_bw.pressed.connect(self.set_range_res)
        layout.addWidget(self.set_bw, 5, 0, 1, 1)
        
        self.quit_button = QPushButton("Quit")
        self.quit_button.pressed.connect(self.end_program)
        layout.addWidget(self.quit_button, 30, 0, 4, 4)
        
        #CFAR Sliders
        self.cfar_bias = QSlider(Qt.Horizontal)
        self.cfar_bias.setMinimum(0)
        self.cfar_bias.setMaximum(100)
        self.cfar_bias.setValue(25)
        self.cfar_bias.setTickInterval(5)
        self.cfar_bias.setMaximumWidth(200)
        self.cfar_bias.setTickPosition(QSlider.TicksBelow)
        self.cfar_bias.valueChanged.connect(self.get_cfar_values)
        layout.addWidget(self.cfar_bias, 8, 0)
        self.cfar_bias_label = QLabel("CFAR Bias (dB): %0.0f" % (self.cfar_bias.value()))
        self.cfar_bias_label.setFont(font)
        self.cfar_bias_label.setAlignment(Qt.AlignLeft)
        self.cfar_bias_label.setMinimumWidth(100)
        self.cfar_bias_label.setMaximumWidth(200)
        layout.addWidget(self.cfar_bias_label, 8, 1)
        
        self.cfar_guard = QSlider(Qt.Horizontal)
        self.cfar_guard.setMinimum(1)
        self.cfar_guard.setMaximum(40)
        self.cfar_guard.setValue(15)
        self.cfar_guard.setTickInterval(4)
        self.cfar_guard.setMaximumWidth(200)
        self.cfar_guard.setTickPosition(QSlider.TicksBelow)
        self.cfar_guard.valueChanged.connect(self.get_cfar_values)
        layout.addWidget(self.cfar_guard, 10, 0)
        self.cfar_guard_label = QLabel("Num Guard Cells: %0.0f" % (self.cfar_guard.value()))
        self.cfar_guard_label.setFont(font)
        self.cfar_guard_label.setAlignment(Qt.AlignLeft)
        self.cfar_guard_label.setMinimumWidth(100)
        self.cfar_guard_label.setMaximumWidth(200)
        layout.addWidget(self.cfar_guard_label, 10, 1)
        
        self.cfar_ref = QSlider(Qt.Horizontal)
        self.cfar_ref.setMinimum(1)
        self.cfar_ref.setMaximum(100)
        self.cfar_ref.setValue(16)
        self.cfar_ref.setTickInterval(10)
        self.cfar_ref.setMaximumWidth(200)
        self.cfar_ref.setTickPosition(QSlider.TicksBelow)
        self.cfar_ref.valueChanged.connect(self.get_cfar_values)
        layout.addWidget(self.cfar_ref, 12, 0)
        self.cfar_ref_label = QLabel("Num Ref Cells: %0.0f" % (self.cfar_ref.value()))
        self.cfar_ref_label.setFont(font)
        self.cfar_ref_label.setAlignment(Qt.AlignLeft)
        self.cfar_ref_label.setMinimumWidth(100)
        self.cfar_ref_label.setMaximumWidth(200)
        layout.addWidget(self.cfar_ref_label, 12, 1)

        # waterfall level slider
        self.low_slider = QSlider(Qt.Horizontal)
        self.low_slider.setMinimum(-100)
        self.low_slider.setMaximum(20)
        self.low_slider.setValue(-100)
        self.low_slider.setTickInterval(5)
        self.low_slider.setMaximumWidth(200)
        self.low_slider.setTickPosition(QSlider.TicksBelow)
        self.low_slider.valueChanged.connect(self.get_water_levels)
        layout.addWidget(self.low_slider, 16, 0)

        self.high_slider = QSlider(Qt.Horizontal)
        self.high_slider.setMinimum(-100)
        self.high_slider.setMaximum(20)
        self.high_slider.setValue(20)
        self.high_slider.setTickInterval(5)
        self.high_slider.setMaximumWidth(200)
        self.high_slider.setTickPosition(QSlider.TicksBelow)
        self.high_slider.valueChanged.connect(self.get_water_levels)
        layout.addWidget(self.high_slider, 18, 0)

        self.water_label = QLabel("Waterfall Intensity Levels")
        self.water_label.setFont(font)
        self.water_label.setAlignment(Qt.AlignCenter)
        self.water_label.setMinimumWidth(100)
        self.water_label.setMaximumWidth(200)
        layout.addWidget(self.water_label, 15, 0,1,1)
        self.low_label = QLabel("LOW LEVEL: %0.0f" % (self.low_slider.value()))
        self.low_label.setFont(font)
        self.low_label.setAlignment(Qt.AlignLeft)
        self.low_label.setMinimumWidth(100)
        self.low_label.setMaximumWidth(200)
        layout.addWidget(self.low_label, 16, 1)
        self.high_label = QLabel("HIGH LEVEL: %0.0f" % (self.high_slider.value()))
        self.high_label.setFont(font)
        self.high_label.setAlignment(Qt.AlignLeft)
        self.high_label.setMinimumWidth(100)
        self.high_label.setMaximumWidth(200)
        layout.addWidget(self.high_label, 18, 1)

        self.steer_slider = QSlider(Qt.Horizontal)
        self.steer_slider.setMinimum(-80)
        self.steer_slider.setMaximum(80)
        self.steer_slider.setValue(0)
        self.steer_slider.setTickInterval(20)
        self.steer_slider.setMaximumWidth(200)
        self.steer_slider.setTickPosition(QSlider.TicksBelow)
        self.steer_slider.valueChanged.connect(self.get_steer_angle)
        layout.addWidget(self.steer_slider, 22, 0)
        self.steer_title = QLabel("Receive Steering Angle")
        self.steer_title.setFont(font)
        self.steer_title.setAlignment(Qt.AlignCenter)
        self.steer_title.setMinimumWidth(100)
        self.steer_title.setMaximumWidth(200)
        layout.addWidget(self.steer_title, 21, 0)
        self.steer_label = QLabel("%0.0f DEG" % (self.steer_slider.value()))
        self.steer_label.setFont(font)
        self.steer_label.setAlignment(Qt.AlignLeft)
        self.steer_label.setMinimumWidth(100)
        self.steer_label.setMaximumWidth(200)
        layout.addWidget(self.steer_label, 22, 1,1,2)

        # FFT plot
        self.fft_plot = pg.plot()
        self.fft_plot.setMinimumWidth(600)
        self.fft_curve = self.fft_plot.plot(freq, pen={'color':'y', 'width':2})
        self.fft_threshold = self.fft_plot.plot(freq, pen={'color':'r', 'width':2})
        title_style = {"size": "20pt"}
        label_style = {"color": "#FFF", "font-size": "14pt"}
        self.fft_plot.setLabel("bottom", text="Frequency", units="Hz", **label_style)
        self.fft_plot.setLabel("left", text="Magnitude", units="dB", **label_style)
        self.fft_plot.setTitle("Received Signal - Frequency Spectrum", **title_style)
        layout.addWidget(self.fft_plot, 0, 2, self.num_rows, 1)
        self.fft_plot.setYRange(-60, 0)
        self.fft_plot.setXRange(signal_freq, signal_freq+plot_freq)

        # Waterfall plot
        self.waterfall = pg.PlotWidget()
        self.imageitem = pg.ImageItem()
        self.waterfall.addItem(self.imageitem)
        # Use a viridis colormap
        pos = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        color = np.array([[68, 1, 84,255], [59, 82, 139,255], [33, 145, 140,255], [94, 201, 98,255], [253, 231, 37,255]], dtype=np.ubyte)
        lut = pg.ColorMap(pos, color).getLookupTable(0.0, 1.0, 256)
        self.imageitem.setLookupTable(lut)
        self.imageitem.setLevels([0,1])
        tr = QtGui.QTransform()
        tr.translate(0,-sample_rate/2)
        tr.scale(0.35, sample_rate / fft_size)
        self.imageitem.setTransform(tr)
        zoom_freq = 35e3
        self.waterfall.setRange(yRange=(signal_freq, signal_freq + zoom_freq))
        self.waterfall.setTitle("Waterfall Spectrum", **title_style)
        self.waterfall.setLabel("left", "Frequency", units="Hz", **label_style)
        self.waterfall.setLabel("bottom", "Time", units="sec", **label_style)
        layout.addWidget(self.waterfall, 0 + self.num_rows + 1, 2, self.num_rows, 1)
        self.img_array = np.ones((num_slices, fft_size))*(-100)

        widget.setLayout(layout)
        # setting this widget as central widget of the main window
        self.setCentralWidget(widget)

    def get_range_res(self):
        """ Updates the slider bar label with Chirp bandwidth and range resolution
		Returns:
			None
		"""
        bw = self.bw_slider.value() * 1e6
        range_res = c / (2 * bw)

    def get_cfar_values(self):
        """ Updates the cfar values
		Returns:
			None
		"""
        self.cfar_bias_label.setText("CFAR Bias (dB): %0.0f" % (self.cfar_bias.value()))
        self.cfar_guard_label.setText("Num Guard Cells: %0.0f" % (self.cfar_guard.value()))
        self.cfar_ref_label.setText("Num Ref Cells: %0.0f" % (self.cfar_ref.value()))


    def get_water_levels(self):
        """ Updates the waterfall intensity levels
		Returns:
			None
		"""
        if self.low_slider.value() > self.high_slider.value():
            self.low_slider.setValue(self.high_slider.value())
        self.low_label.setText("LOW LEVEL: %0.0f" % (self.low_slider.value()))
        self.high_label.setText("HIGH LEVEL: %0.0f" % (self.high_slider.value()))

    def get_steer_angle(self):
        """ Updates the steering angle readout
		"""
        self.steer_label.setText("%0.0f DEG" % (self.steer_slider.value()))
        phase_delta = (2 * 3.14159 * output_freq * my_phaser.element_spacing
            * np.sin(np.radians(self.steer_slider.value()))
            / (3e8)
        )
        my_phaser.set_beam_phase_diff(np.degrees(phase_delta))

    def set_range_res(self):
        """ Sets the Chirp bandwidth
		Returns:
			None
		"""
        global dist, slope, signal_freq, plot_freq
        bw = self.bw_slider.value() * 1e6
        slope = bw / ramp_time_s
        dist = (freq - signal_freq) * c / (2 * slope)
        my_phaser.freq_dev_range = int(bw / 4)  # frequency deviation range in Hz
        my_phaser.enable = 0

    def end_program(self):
        """ Gracefully shutsdown the program and Pluto
		"""
        my_sdr.tx_destroy_buffer()
        print("Program finished and Pluto Tx Buffer Cleared")
        # disable TDD and revert to non-TDD (standard) mode
        tdd.enable = False
        sdr_pins.gpio_phaser_enable = False
        tdd.channel[1].polarity = not(sdr_pins.gpio_phaser_enable)
        tdd.channel[2].polarity = sdr_pins.gpio_phaser_enable
        tdd.enable = True
        tdd.enable = False
        self.close()

    def change_thresh(self, state):
        """ Toggles between showing cfar threshold values
		Args:
			state (QtCore.Qt.Checked) : State of check box
		Returns:
			None
		"""
        global plot_threshold
        plot_state = win.fft_plot.getViewBox().state
        if state == QtCore.Qt.Checked:
            plot_threshold = True
        else:
            plot_threshold = False

    def change_cfar(self, state):
        """ Toggles between enabling/disabling CFAR
		Args:
			state (QtCore.Qt.Checked) : State of check box
		Returns:
			None
		"""
        global cfar_toggle
        if state == QtCore.Qt.Checked:
            cfar_toggle = True
        else:
            cfar_toggle = False

# create pyqt5 app
App = QApplication(sys.argv)

# create the instance of our Window
win = Window()
index = 0

def update():
    """ Updates the FFT in the window
	"""
    global index, plot_threshold, freq, dist, plot_dist, ramp_time_s, sample_rate
    label_style = {"color": "#FFF", "font-size": "14pt"}
    my_phaser._gpios.gpio_burst = 0
    my_phaser._gpios.gpio_burst = 1
    my_phaser._gpios.gpio_burst = 0
    data = my_sdr.rx()
    chan1 = data[0]
    chan2 = data[1]
    sum_data = chan1+chan2
    
    # select just the linear portion of the last chirp
    rx_bursts = np.zeros((num_chirps, good_ramp_samples), dtype=complex)
    for burst in range(num_chirps):
        start_index = start_offset_samples + burst*num_samples_frame
        stop_index = start_index + good_ramp_samples
        rx_bursts[burst] = sum_data[start_index:stop_index]
        burst_data = np.ones(fft_size, dtype=complex)*1e-10
        #win_funct = np.blackman(len(rx_bursts[burst]))
        win_funct = np.ones(len(rx_bursts[burst]))
        burst_data[start_offset_samples:(start_offset_samples+good_ramp_samples)] = rx_bursts[burst]*win_funct

    sp = np.absolute(np.fft.fft(burst_data))
    sp = np.fft.fftshift(sp)
    s_mag = np.abs(sp) / np.sum(win_funct)
    s_mag = np.maximum(s_mag, 10 ** (-15))
    s_dbfs = 20 * np.log10(s_mag / (2 ** 11))

    bias = win.cfar_bias.value()
    num_guard_cells = win.cfar_guard.value()
    num_ref_cells = win.cfar_ref.value()
    cfar_method = 'average'
    if (True):
        threshold, targets = cfar(s_dbfs, num_guard_cells, num_ref_cells, bias, cfar_method)
        s_dbfs_cfar = targets.filled(-200)  # fill the values below the threshold with -200 dBFS
        s_dbfs_threshold = threshold

    win.fft_threshold.setData(freq, s_dbfs_threshold)
    if plot_threshold:
        win.fft_threshold.setVisible(True)
    else:
        win.fft_threshold.setVisible(False)

    win.img_array = np.roll(win.img_array, 1, axis=0)
    if cfar_toggle:
        win.fft_curve.setData(freq, s_dbfs_cfar)
        win.img_array[0] = s_dbfs_cfar
    else:
        win.fft_curve.setData(freq, s_dbfs)
        win.img_array[0] = s_dbfs
    win.imageitem.setLevels([win.low_slider.value(), win.high_slider.value()])
    win.imageitem.setImage(win.img_array, autoLevels=False)

    if index == 1:
        win.fft_plot.enableAutoRange("xy", False)
    index = index + 1

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)

# start the app
sys.exit(App.exec())
