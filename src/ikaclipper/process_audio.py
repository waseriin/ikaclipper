#--------------------------------------------------
# Process STFT data as it's being parsed
# https://librosa.github.io/librosa/_modules/librosa/core/spectrum.html#stft
#--------------------------------------------------

import datetime
import os
import subprocess
import sys

import numpy as np
import scipy
import scipy.signal

from . import process_file as pf

def main(t_source:tuple, markers:list, sample_rate:int=None, config=None):
    l_markers = load_markers(markers, sample_rate, config)

    source, track = t_source
    d_source = load_source(source, track, sample_rate, config)

    return _get_matches(d_source, l_markers, sample_rate, config)


def load_sample(filepath:str, track:int, sample_rate:int=None, config=None):
    """ process input with audio, returns a dictionary consits of stft results.

    Args:
        filepath (str): Expects verified with `pf.set_file()`.
        track (int): Expects verified with `pf.get_acodec()`.
        sample_rate (int, optional): 
        config (_type_, optional): 

    Returns:
        dict: _description_
    """
    ''' loads an audio file of `filepath`, then returns a dictionary consists of stft results. filepath is required.
    '''
    if sample_rate is None:
        sample_rate = pf.get_option_int('sample_rate', item='audio_params', config=config)

    sample_pcm = _generate_pcm_data(filepath, track, sample_rate, config)

    # sample_duration = round(float(len(sample_pcm)) / sample_rate, 2)
    # print(f' loaded {filepath} ({sample_duration} s)')

    d_sample = _stft_raw(sample_pcm, config=config)
    d_sample['filepath'] = os.path.abspath(filepath)
    d_sample['audio_track'] = track

    return d_sample


def load_samples(l_samples:list, sample_rate:int=None, config=None):
    """load multiple samples, then returns processed result as a list.

    Args:
        l_samples (list): a list of tuples. a tuple consists of a filepath and an audio track to be used.
        sample_rate (int, optional):
        config (_type_, optional):

    Returns:
        list: a list of d_sample. The length of the list matches the number of samples.
    """
    if sample_rate is None:
        sample_rate = pf.get_option_int('sample_rate', item='audio_params', config=config)

    samples = []
    for filepath, track in l_samples:
        samples.append(load_sample(filepath, track, sample_rate, config))

    return samples


def load_source(source:str, track:int, sample_rate:int=None, config=None):
    """ takes valid audio track, 

    Args:
        source (str): a path to the source file.
        track (int): an audio track of the source to be used.
        sample_rate (int, optional): _description_. Defaults to None.
        config (_type_, optional): _description_. Defaults to None.

    Returns:
        dict: a d_sample for the source.
    """
    print(f'Loading source file: track #{track} of {source}')

    return load_sample(source, track, sample_rate, config)


def load_markers(markers:list, sample_rate:int=None, config=None):
    print('Loading marker files..')
    n_fft = pf.get_option_int('n_fft', item='audio_params', config=config)
    d_type = pf.get_option_str('d_type', item='audio_params', config=config)
    sample_crop_start = pf.get_option_int('sample_crop_start', item='audio_params', config=config)
    sample_crop_end = pf.get_option_int('sample_crop_end', item='audio_params', config=config)

    marker_samples = load_samples(markers, sample_rate=sample_rate, config=config)
    l_markers = []

    for sample in marker_samples:
        filepath, sample_frames, fft_window, n_columns = sample['filepath'], sample['frames'], sample['fft_window'], sample['n_columns']

        # Pre-allocate the STFT matrix
        sample_data = np.empty((int(1 + n_fft // 2), sample_frames.shape[1]), dtype=d_type, order='F')

        for bl_s in range(0, sample_data.shape[1], n_columns):
            bl_t = min(bl_s + n_columns, sample_data.shape[1])
            sample_data[:, bl_s:bl_t] = scipy.fft.fft(fft_window * sample_frames[:, bl_s:bl_t], axis=0)[:sample_data.shape[0]]

        sample_data = abs(sample_data)

        sample_height = sample_data.shape[0]
        sample_length = sample_data.shape[1]

        x = 0
        sample_start = 0
        while x < sample_length:
            total = 0
            for y in range(0, sample_height):
                total += sample_data[y][x]
            if total >= 1:
                sample_start = x
                break
            x += 1
        sample_start += sample_crop_start # The first few frames seem to get modified, perhaps due to compression?
        sample_end = (sample_length - sample_crop_end)

        l_markers.append({
                'sample_start': sample_start,
                'sample_end': sample_end,
                'filepath': os.path.basename(filepath),
                'sample_data': sample_data
            })

        print('  {} ({}/{})'.format(filepath, sample_start, sample_end))
    return l_markers


def _get_matches(d_source:dict, l_markers:list, sample_rate:int=None, config=None, max=-1):

    print('Processing ..')

    d_type = pf.get_option_str('d_type', item='audio_params', config=config)
    hz_count = pf.get_option_int('hz_count', item='audio_params', config=config)
    hop_length = pf.get_option_int('hop_length', item='audio_params', config=config)

    sample_warn_allowance = pf.get_option_int('sample_warn_allowance', item='audio_params', config=config)
    match_any_sample = pf.get_option_bool('match_any_sample', item='audio_params', config=config)
    matching_min_score = pf.get_option_float('matching_min_score', item='audio_params', config=config)
    matching_skip = pf.get_option_int('matching_skip', item='audio_params', config=config)
    matching_ignore = pf.get_option_int('matching_ignore', item='audio_params', config=config)

    if sample_rate is None:
        sample_rate = pf.get_option_int('sample_rate', item='audio_params', config=config)

    samples = l_markers
    
    source_frame_start = pf.get_option_int('source_frame_start', item='audio_params', config=config)
    source_frame_end = pf.get_option_int('source_frame_end', item='audio_params', config=config)

    if source_frame_end is None:
        source_frame_end = d_source.get('frames').shape[1]

    print(f'Find from {source_frame_start} to {source_frame_end}')

    matching = {}
    match_count = 0
    match_last_time = None
    match_last_ignored = False
    match_skipping = 0
    d_matches = {
        'source': d_source['filepath'],
        'audio_track': d_source['audio_track']
        }

    results_end = {}
    results_dupe = {}
    for sample_id, sample_info in enumerate(samples):
        results_end[sample_id] = {}
        results_dupe[sample_id] = {}
        for k in range(0, (sample_info['sample_end'] + 1)):
            results_end[sample_id][k] = 0
            results_dupe[sample_id][k] = 0

    for block_start in range(source_frame_start, source_frame_end, d_source.get('n_columns')): # Time in 31 blocks

        block_end = min(block_start + d_source.get('n_columns'), source_frame_end)

        set_data = abs((scipy.fft.fft(d_source.get('fft_window') * d_source.get('frames')[:, block_start:block_end], axis=0)).astype(d_type))

        # print('  {} to {} - {}'.format(block_start, block_end, str(datetime.timedelta(seconds=((float(block_start) * hop_length) / sample_rate)))))

        x = 0
        x_max = (block_end - block_start)
        while x < x_max:

            if match_skipping > 0:
                if x == 0:
                    print('    Skipping {}'.format(match_skipping))
                match_skipping -= 1
                x += 1
                continue

            matching_complete = []
            for matching_id in list(matching): # Continue to check matches (i.e. have already started)

                sample_id = matching[matching_id][0]
                sample_x = (matching[matching_id][1] + 1)

                if sample_id in matching_complete:
                    continue

    # TEST-2... this is the main test (done after the first frame has been matched with TEST-1)

                ###
                # While this does not work, maybe we could try something like this?
                #
                #     match_min_score = (0 - config['matching_min_score']);
                #
                #     hz_score = (set_data[0:hz_count,x] - samples[sample_id][3][0:hz_count,sample_x])
                #     hz_score = (hz_score < match_min_score).sum()
                #
                #     if hz_score < 5:
                #
                ###
                # Correlation might work better, but I've no idea how to use it.
                #   np.correlate(set_data[0:hz_count,x], sample_info[3][0:hz_count,sample_start])[0]
                ###

                # Return a list of Hz buckets for this frame (set_data[0-1025][x]),
                # This is where `hz_score` starts as a simple array, using a column of results at time position `x`.
                # Subtract them all from the equivalent Hz bucket from sample_start (frame 0, ish)
                # Convert to positive values (abs),
                # Calculate the average variation, as a float (total/count).

                hz_score = abs(set_data[0:hz_count,x] - samples[sample_id]['sample_data'][0:hz_count,sample_x])
                hz_score = sum(hz_score)/float(len(hz_score))

                if hz_score < matching_min_score:

                    if sample_x >= samples[sample_id]['sample_end']:

                        match_start_time = ((float(x + block_start - samples[sample_id]['sample_end']) * hop_length) / sample_rate)

                        print(f' Found @ {str(datetime.timedelta(seconds=match_start_time)).split(".")[0]} - {samples[sample_id]["filepath"]}')

                        results_end[sample_id][sample_x] += 1

                        # if (matching_skip) or (match_last_time == None) or ((match_start_time - match_last_time) > matching_ignore):
                        #     match_last_ignored = False
                        # else:
                        #     match_last_ignored = True

                        # TODO: take a close look here
                        # matches.append(sample_id, match_start_time, match_last_ignored])
                        # matches.setdefault(samples[sample_id]['filepath'], []).append([match_start_time, match_last_ignored])
                        d_matches.setdefault(samples[sample_id]['filepath'], []).append(match_start_time)
                        # match_last_time = match_start_time

                        if matching_skip:
                            match_skipping = ((matching_skip * sample_rate) / hop_length)
                            print('    Skipping {}'.format(match_skipping))
                            matching = {}
                            break # No more 'matching' entires
                        else:
                            del matching[matching_id]
                            matching_complete.append(sample_id)

                    else:

                        # print('    Match {}/{}: Update to {} ({} < {})'.format(matching_id, sample_id, sample_x, hz_score, matching_min_score))
                        matching[matching_id][1] = sample_x

                elif matching[matching_id][2] < sample_warn_allowance and sample_x > 10:

                    # print('    Match {}/{}: Warned at {} of {} ({} > {})'.format(matching_id, sample_id, sample_x, samples[sample_id]['sample_end'], hz_score, matching_min_score))
                    matching[matching_id][2] += 1

                else:

                    # print('    Match {}/{}: Failed at {} of {} ({} > {})'.format(matching_id, sample_id, sample_x, samples[sample_id]['sample_end'], hz_score, matching_min_score))
                    results_end[sample_id][sample_x] += 1
                    del matching[matching_id]

            if match_skipping > 0:
                continue

            for matching_sample_id in matching_complete:
                for matching_id in list(matching):
                    if match_any_sample or matching[matching_id][0] == matching_sample_id:
                        sample_id = matching[matching_id][0]
                        sample_x = matching[matching_id][1]
                        # print('    Match {}/{}: Duplicate Complete at {}'.format(matching_id, sample_id, sample_x))
                        results_dupe[sample_id][sample_x] += 1
                        del matching[matching_id] # Cannot be done in the first loop (next to continue), as the order in a dictionary is undefined, so you could have a match that started later, getting tested first.

            for sample_id, sample_info in enumerate(samples): # For each sample, see if the first frame (after sample_crop_start), matches well enough to keep checking (that part is done above).

                sample_start = sample_info['sample_start']

    # TEST-1

                hz_score = abs(set_data[0:hz_count,x] - sample_info['sample_data'][0:hz_count,sample_start])
                hz_score = sum(hz_score)/float(len(hz_score))

                if hz_score < matching_min_score:

                    match_count += 1
                    # print('    Match {}: Start for sample {} at {} ({} < {})'.format(match_count, sample_id, (x + block_start), hz_score, matching_min_score))
                    matching[match_count] = [
                            sample_id,
                            sample_start,
                            0, # Warnings
                        ]

            x += 1

    #--------------------------------------------------

    print('\nMatches')
    for key, timestamps in d_matches.items():
        if "." in key:
            print(f' Found {key} for {len(timestamps)} times')
    return d_matches


def _generate_pcm_data(sample_path:str, track:int, sample_rate:int, config=None):

    ffmpeg_path = pf.get_option_str('ffmpeg_path', 'path', config)
    achannels = pf.get_option_str('audio_channels', 'audio_params', config)

    # if audio_track is None:
    #     audio_track = pf.get_option_str('default_audio_track', item='params', config=config)

    # if sample_rate is None:
    #     sample_rate = pf.get_option_int('sample_rate', item='audio_params', config=config)

    command_list = [ffmpeg_path, '-vn', '-loglevel', 'error', '-stats', 
                    '-i', sample_path, 
                    '-f', 's16le', '-ac', achannels,
                    '-map', f'0:a:{track}', '-ar', str(sample_rate), '-']

    proc = subprocess.run(command_list, capture_output=True)

    scale = 1./float(1 << ((8 * 2) - 1))
    pcm_data = scale * np.frombuffer(proc.stdout, '<i2').astype(np.float32)

    if pcm_data.size == 0:
        print(f'  Missing data from Audio file "{sample_path}" via "{ffmpeg_path}"')
        sys.exit()

    return pcm_data


def _stft_raw(series, config=None):
    n_fft = pf.get_option_int('n_fft', item='audio_params', config=config)
    win_length = pf.get_option_int('win_length', item='audio_params', config=config)
    hop_length = pf.get_option_int('hop_length', item='audio_params', config=config)
    hz_count = pf.get_option_int('hz_count', item='audio_params', config=config)
    d_type = pf.get_option_str('d_type', item='audio_params', config=config)

    #--------------------------------------------------
    # Config

    window = 'hann'
    pad_mode='reflect'

    #--------------------------------------------------
    # Get Window

    fft_window = scipy.signal.get_window(window, win_length, fftbins=True)

    #--------------------------------------------------
    # Pad the window out to n_fft size... Wrapper for
    # np.pad to automatically centre an array prior to padding.

    axis = -1

    n = fft_window.shape[axis]

    lpad = int((n_fft - n) // 2)

    lengths = [(0, 0)] * fft_window.ndim
    lengths[axis] = (lpad, int(n_fft - n - lpad))

    if lpad < 0:
        raise ValueError(('Target size ({:d}) must be at least input size ({:d})').format(n_fft, n))

    fft_window = np.pad(fft_window, lengths, mode='constant')

    #--------------------------------------------------
    # Reshape so that the window can be broadcast

    fft_window = fft_window.reshape((-1, 1))

    #--------------------------------------------------
    # Pad the time series so that frames are centred

    series = np.pad(series, int(n_fft // 2), mode=pad_mode)

    #--------------------------------------------------
    # Window the time series.

        # Compute the number of frames that will fit. The end may get truncated.
    frame_count = 1 + int((len(series) - n_fft) / hop_length) # Where n_fft = frame_length

        # Vertical stride is one sample
        # Horizontal stride is `hop_length` samples
    frames_data = np.lib.stride_tricks.as_strided(series, shape=(n_fft, frame_count), strides=(series.itemsize, hop_length * series.itemsize))

    #--------------------------------------------------
    # how many columns can we fit within MAX_MEM_BLOCK

    MAX_MEM_BLOCK = 2**8 * 2**10
    n_columns = int(MAX_MEM_BLOCK / (hz_count * (np.array(0, dtype=d_type).itemsize)))

    #--------------------------------------------------
    # Return
    d_series = {
        # 'series': series,
        'frames': frames_data,
        'fft_window': fft_window,
        'n_columns': n_columns
    }

    return d_series
