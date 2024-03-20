%% EGB242 Assignment 1 %%
% This file is a template for your MATLAB solution.
%
% Before starting to write code, record your test audio with the record
% function as described in the assignment task.
%% Load recorded test audio into workspace
clear all;
close all;
load DataA1;
% Begin writing your MATLAB solution below this line.
%1.3
%{
Create a time vector t for the audio signal. Plot the audio signal against t.
Comment on
your observations, and how they relate to any audible characteristics of the
signal.
%}
N = length(audio);
%Number of samples
AudioInSeconds=(length(audio)/fs);
%Audio length in seconds
t = linspace(0, AudioInSeconds, N);
%Time vector
%1.5
%{
Using negative 5 to 5 (i.e., 5 harmonics), generate a vector cn in
MATLAB which contains cn evaluated at each value of n. List the values
of these coefficients in your report.
%}
harmonic = -5:5;
cn2 = generate_cn(harmonic);
%1.6
%{
Using the cn vector, generate an approximation of the noise signal nApprox for
the full
time vector t. Plot your recorded audio and your generated noise signal
approximation.
%}
f0 = 1/2;
nApprox = ApproximationValues(harmonic, cn2, t, f0);
% Generate the noise approximation
% Plot recorded audio and noise signal approximation
figure;
plot_signal(1, t, audio, 'Time (s)', 'Amplitude', 'Recorded audio');
plot_signal(2, t, real(nApprox), 'Time (s)', 'Amplitude', 'Noiseapproximation');
%1.7
%{
De-noise the recorded audio by reversing the additive noise process (Figure 2)
using your
Fourier series approximation, and store the de-noised signal in audioClean.
Listen to
the clean signal, and plot it.
%}
audioClean = audio - nApprox;
%Remove noise
audioClean = real(audioClean);
%Unable to use sound function without real
% Plot clean signal
plot_signal(3, t, audioClean, 'Time (s)', 'Amplitude', 'Clean Audio Signal:(s)');
%1.8
%{
Is using 5 harmonics in your noise signal approximation enough to adequately
de-noise
the audio? Experiment with the number of harmonics to determine a suitable
value, and
justify your choice both qualitatively and quantitatively.
%}
% I believe 5 to be a sufficent number. Changing the values to 10:-10 did
% not provide much benefit apart from the audio sounding clearer by a small
% margin. This compared to the computational time it took to undertake
% reveals that 5 to -5 harmonics to be a sufficient choice in this
% situation.
%2.1
%{
Plot the magnitude spectrum of the clean audio signal, using an appropriate
frequency
vector f
%}
%Frequency domain for audio clean
N = length(audioClean); %Number of samples
frequencyVector = (0:N-1)*(fs/N); %Frequency vector
fftAudioClean = fft(audioClean); %Fast Fourier Transform of audio clean
fftAudioClean = abs(fftAudioClean); %Absolute values only of fftAudioClean
plot_signal(4, frequencyVector, fftAudioClean, 'Frequency (Hz)', 'Magnitude','Clean Audio: (Hz)');
%2.2
%{
Listen to the channel before transmitting your signal through it. You
can listen to the channel before transmitting anything by passing a vector
of zeroes through the channel function.
%}
channelQuiet = channel(10541977, zeros(size(t))); %Generate channel
sound(channelQuiet, fs);
%Play channel
%The channelQuiet is mostly just static noise. It has high frequency
% blips that discern itself from the rest of the static, but i cannot
% intrepolate any information from this sound.
%2.3
%{
Plot the time and frequency domain of channelQuiet in order to find an
empty band of frequencies that you can transmit your audio on. State
your selected range of frequencies and the center frequency. Justify these
parameter choices.
%}
%Time domain for channel
plot_signal(5, t, channelQuiet, 'Time (s)', 'Amplitude', 'Channel Pre-Transmission');
%Frequency domain for channel
% Set x-axis limits to 0 to 2
fftChannelQuiet = abs(fft(channelQuiet)); %Absolute values only of fftChannelQuiet
plot_signal(6, frequencyVector, fftChannelQuiet, 'Frequency (Hz)', 'Magnitude','Channel Pre-Transmission');
%2.4
%{
Modulate your audio signal using the carrier frequency you have selected.
%}
%Begin Modulation
fc = 60000; %Carrier Frequency
AudioModulated = cos(2*pi*fc*t) .* audioClean;
%Modulate audio signal
%2.5
%{
Simulate the transmission of your modulated signal, providing it as input to
the channel
function, and plot the frequency domain of the input and output signals.
%}
AudioModulated_FFT = fft(AudioModulated); %Fast Fourier Transform of audio modulated
fVec = linspace(-fs/2, fs/2, length(t)); %Frequency vector
plot_signal(7, fVec, abs(AudioModulated_FFT), 'Frequency (Hz)', 'Magnitude','AudioModulated Input');
AudioModulatedOutput = channel(10541977, AudioModulated); %Modulated Output
AudioModulatedOutput_FFT = fft(AudioModulatedOutput); %Fast Fourier Transform of audio modulated output
plot_signal(8, fVec, abs(AudioModulatedOutput_FFT), 'Frequency (Hz)','Magnitude', 'AudioModulated Output');
%2.6
%{
Demodulate your audio signal from the channel output created in 2.5. View the
demodu-
lated signal in the frequency domain. Filter the demodulated signal to isolate
your audio
signal. Use the lowpass function in MATLAB to simulate an analogue filter, and
store
the received audio as audioReceived.
%}
AudioDeModulated = AudioModulatedOutput .* cos(2*pi*fc*t); %Demodulate audio signal
f_demod = linspace(-fs/2, fs/2, length(AudioDeModulated)); %Frequency vector
AudioDeModulated_FFT = fft(AudioDeModulated); %Fast Fourier Transform of audio demodulated
plot_signal(9, f_demod, abs(AudioDeModulated_FFT), 'Frequency (Hz)','Magnitude', 'AudioDeModulated');
%FILTER
audioReceived = lowpass(AudioDeModulated, 100, fc); %Filter audio signal
audioReceived = real(audioReceived); %Remove imaginary component
audioReceived = audioReceived / max(abs(audioReceived)); %Normalise audio signal
plot_signal(10, t, audioReceived, 'Time (s)', 'Amplitude', 'Filtered + Demod');
%3.1
fs2 = 48000; %New sampling frequency
audioResampled = resample(audioReceived, fs2, fs); %Resample audio signal
%3.2
%{
Listen to and comment on the resampled audio.
%}
sound(audioResampled, fs2);
%Play audio resampled
%The audio is very clear. It has little-to-none clipping
%Though the high frequency blips at even integers still persists.
%3.3
%{
With an appropriate quantiser (mid-tread or mid-riser), quantise audioResampled
using 16 quantisation levels and store the result as audioQuantised. Listen to
and
plot the quantised audio and comment on any changes.
%}
quantise_plot(16, audioResampled, fs2, 11);
quantise_plot(2, audioResampled, fs2, 13);
quantise_plot(4, audioResampled, fs2, 15);
quantise_plot(8, audioResampled, fs2, 17);
audioQuantised = quantise_plot(32, audioResampled, fs2, 19);
%From the graphical views of each quantization process, the
%relationship of increasing quantization levels and increase of audio
%quality can cleary be seen. This is evident in how the mid-riser with
%2 levels looks similair to a barcode, barely representing the audio
%signal at all, but at 32 levels can almost identically represent the
%audio. The final choice for quantization levels will be 32, simply
%because it most accurately represents the audio signal with minimal
%noise. The choice of quantizer will be mid-tread due to its dead zone
%nature around zero. This means that small signal close to zero are
%quantized to zero. This is important because audio signals are
%low-amplitude content. It can help reduce the noise by forcing
%low-amplitude signals to zero.

function plot_signal(subplot_position, x_data, y_data, x_label, y_label,title_text)
subplot(4,6,subplot_position);
plot(x_data, y_data);
xlabel(x_label);
ylabel(y_label);
title(title_text);
end


function ApproxVector= ApproximationValues(HarmonicVector, cnVector, t, f0)
nApprox = zeros(size(t)); % Initialise approximation vector
for i = 1:length(HarmonicVector) % Loop through harmonics
n = HarmonicVector(i); % Set current harmonic
nApprox = nApprox + cnVector(i)*exp(1j*2*pi*n*f0*t); % Add value to
nApprox;
end
ApproxVector = nApprox; % Set output
end

function [cn2] = generate_cn(harmonicVector)
% Calculate C0
C0 = (19-5*exp(-4))/8; %The handcalculated expression for C0
% Calculate Cn for -5 <= n <= 5
cn2 = zeros(size(harmonicVector)); %Create a vector of zeros to store the values of cn the size of harmonic (1x11)
for i = 1:length(harmonicVector) %For each value of n in harmonic
n = harmonicVector(i); %Set n to the current value of harmonic
CnthValue = 0.5*((5*(exp(4-1j*pi*n)-1))./(exp(4)*(4-1j*pi*n)) - 3*((-2*exp(-2*1j*pi*n)+exp(-1j*pi*n))./(1j*pi*n) +(-exp(-2*1j*pi*n)+exp(-1j*pi*n))./(1j^2*pi^2*n^2)) - 8*(exp(-2*1j*pi*n)-exp(-1j*pi*n))./(1j*pi*n)); %The handcalculated expression for Cn
cn2(i) = CnthValue; %Save it to the vector
end
%cn2(6) = C0; %C0 is the 6th value in the vector, so we need to save it there.
%this value will be different for each harmonic range
%Itll always be the max number + 1
%For harmonic range of -5:5 its 5+1 = 6
%For harmonic range of -20:20 its 20+1 = 21
%Theres probably a bettwer way to do this but its working, so.
cn2(max(harmonicVector)+1)=C0;
end
function [audioMT] = quantise_plot(L, audioResampled, fs2, subplotPos)
% Initialize variables
xmax = 1; xmin = -1;
N = length(audioResampled);
AudioInSeconds = (length(audioResampled) / fs2);
t = linspace(0, AudioInSeconds, N);
deltaMR = (xmax-xmin)/L;
deltaMT = (xmax-xmin/L-1);
% Calculate Mid-Tread and Mid-Riser quantised audio
audioMT = deltaMT * floor(audioResampled/deltaMT + 1/2);
audioMR = deltaMR * (floor((audioResampled - xmin) / deltaMR) + 1/2) + xmin;
% Apply limits
audioMT(audioMT>=xmax) = xmin+deltaMT*(L-1);
audioMR(audioMR>=xmax) = xmin+deltaMR*(L-1/2);
% Plot Mid-Riser
subplot(4, 6, subplotPos);
hold on; grid on;
plot(t, audioMR);
plot(t, audioResampled);
xlabel('Time (s)');
ylabel('Amplitude');
title(['Mid-Riser: N=' num2str(L)]);
% Plot Mid-Tread
subplot(4, 6, subplotPos+1);
hold on; grid on;
plot(t, audioMT);
plot(t, audioResampled);
xlabel('Time (s)');
ylabel('Amplitude');
title(['Mid-Tread: N=' num2str(L)]);
end