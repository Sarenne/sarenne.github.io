####################################################################
# Script to extract to deindentify audio, based on Weston et al 2020
# Takes a list of absolute paths to wavs, adapt if needed.
# Created: Johannah O'Mahony CSTR (2021)
####################################################################

import soundfile as sf
import numpy as np
import librosa
import pandas
import tqdm as tqdm
import parselmouth
from parselmouth.praat import call
from parselmouth import PraatError
import os
import math
import glob

outdir = '/afs/inf.ed.ac.uk/user/s21/s2132904/' #output parent dir
indir = '/disk/scratch2/jomahony/Spotify-Question-Answers/audio_denoised/0/0/show_00BUDdSn801kmcWw50Us8S/' #path wavs


def normalise_wav(wav, output):
    '''Normalises mean/unit variance
    following Weston et al --> don't use'''
    path, base = os.path.split(output)
    os.makedirs(path, exist_ok=True)

    y, sr = librosa.load(wav, sr=800)
    y = (y - y.mean())/ y.std()
    sf.write(output, y, 800)

    return wav


def get_speaker_limits(pitch_values):
    '''Based on deLooze paper, finds optimal pitch floor
    and pitch ceiling per audio file'''
    
    q35 = np.nanquantile(pitch_values, .35)
    q65 = np.nanquantile(pitch_values, .65)
    pfloor = q35 * 0.72 - 10 
    pceiling = q65 * 1.9 + 10

    return pfloor, pceiling


def get_semitones_shift(median):
    '''Returns amount needed to shift
    pitch in semitones for use in praat
    pitch shifted to ref of 100Hz'''

    semi_shift = -1 * (12 * (math.log2(median/100)))

    return semi_shift


def shift_pitch(wav, output):
    '''Shifts pitch to median of 100z for all
    speakers and resynthesises '''

    path, base = os.path.split(output)
    os.makedirs(path, exist_ok=True)
    
    # load file
    sound = parselmouth.Sound(wav)

    # first pass pitch track standard values
    pitch = parselmouth.praat.call(sound,"To Pitch (ac)...", 0.01, 75, 15, "off", 0.03, 0.45, 0.01, 0.35, 0.14, 600)
    pitch_values = pitch.selected_array['frequency']
    pitch_values = pitch_values[pitch_values != 0.0]

    # get optimal floor and ceiling
    pfloor, pceiling = get_speaker_limits(pitch_values)

    # second pass with optimal floor and ceiling
    pitch = parselmouth.praat.call(sound,"To Pitch (ac)...", 0.01, pfloor, 15, "off", 0.03, 0.45, 0.01, 0.35, 0.14, pceiling)
    pitch_values = pitch.selected_array['frequency']
    pitch_values = pitch_values[pitch_values != 0.0]
    median = np.nanmedian(pitch_values)
    semi_shift = get_semitones_shift(median)

    duration = parselmouth.praat.call(sound, "Get end time")

    # shift pitch and resynthesise audio 
    manipulation = call(sound, "To Manipulation", 0.01, 75, 600)
    pitch_tier = call(manipulation, "Extract pitch tier")
    call(pitch_tier, "Shift frequencies", 0.0, duration, semi_shift, 'Semitones')
    call([pitch_tier, manipulation], "Replace pitch tier")

    shifted = call(manipulation, "Get resynthesis (overlap-add)")
    shifted.save(output, 'WAV')

    return


def downsample_audio(wav, outfile):
    '''Downsamples to 800 from 22kHz'''

    path, base = os.path.split(outfile)
    os.makedirs(path, exist_ok=True)
    y, sr = librosa.load(wav, sr=22050)
    y_800 = librosa.resample(y, orig_sr=sr, target_sr=800)
    sf.write(outfile, y_800, 800)

    return


if __name__ == "__main__":

    logfile = open("shift_frequency_errors.log", "w+")

    #Create output paths
    for wav in tqdm.tqdm(glob.glob(indir + "/*.wav")):
        path, base = os.path.split(wav)     
        shift_path = outdir + '/denoised_22khz_mono_16_norm_shift/' + base
        ds_path = outdir + '/denoised_800hz_mono_16_norm/' + base
        ds_shift_path = outdir + '/denoised_800hz_mono_16_norm_shift/' + base

        #try statement because sometimes there are pitch track errors
        try:
            shift_pitch(wav, shift_path)
        except parselmouth.PraatError:
            logfile.write("Error processing {}...\n".format(wav))
            continue 

        downsample_audio(wav, ds_path) #downsample original file without pitch shifting
        downsample_audio(shift_path, ds_shift_path) #downsample shifted