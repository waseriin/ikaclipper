import configparser
import json
import mimetypes
import os
import subprocess
import sys
from tkinter import filedialog, Tk

def get_file_suffix(filename:str):
    return os.path.splitext(filename)[-1]


def get_filename_without_suffix(filename:str):
    return os.path.splitext(os.path.basename(filename))[0]


def get_userconfig():
    filename = 'config.ini'
    cwd = os.getcwd()

    configpath = os.path.join(cwd, filename)
    if os.path.isfile(configpath):
        # print(f"1: Found {filename} in {os.path.dirname(configpath)}")
        return configpath

    home = os.path.expanduser("~")
    configpath = os.path.join(home, 'ikaclipper', filename)
    if os.path.isfile(configpath):
        # print(f"2: Found {filename} in {os.path.dirname(configpath)}")
        return configpath

    configpath = os.path.join(home, '.config', 'ikaclipper', filename)
    if os.path.isfile(configpath):
        # print(f"3: Found {filename} in {os.path.dirname(configpath)}")
        return configpath
    else:
        # print(f"4: Couldn't find {filename}")
        print(f"Continue with default parameters (unless overridden).\nUse of config.ini is recommended.\n")
        return ''


def load_config(user_config:str=None):
    ''' loads config file and merges with the default one

    Args:
        user_config (str, optional): the config file. Defaults to 'config.ini'.

    Returns:
        configparser: merged config values
    '''

    config = configparser.ConfigParser(inline_comment_prefixes=(';', '#'))

    default_config = os.path.join(os.path.dirname(__file__), 'default_config.ini')
    if not os.path.exists(default_config):
        print("Error: Couldn't find default_config.ini")
        sys.exit()
    else:
        config.read([default_config])
        # print('0: default_config.ini has been loaded.')

    if not user_config:
        user_config = get_userconfig()

    if os.path.exists(user_config):
        print(f"Loading {os.path.basename(user_config)} from {os.path.dirname(user_config)} ..")
        config.read([user_config])

    return config


def get_option_str(option:str, item:str='DEFAULT', config:configparser=None):
    if config is None:
        config = load_config()

    opt = config.get(item, option)
    opt = None if opt == "None" else opt

    if opt is not None:
        return opt.strip('"').strip("'")
    else:
        return None


def get_option_int(option:str, item:str='DEFAULT', config:configparser=None):
    # some options requires additional calculation
    if option in ['hz_count', 'win_length', 'hop_length']:
        n_fft = get_option_int('n_fft', item, config)
        if option == 'hz_count':
            return int(1 + n_fft // 2)
        elif option == 'win_length':
            return n_fft
        elif option == 'hop_length':
            win_length = get_option_int('win_length', item, config)
            return int(win_length // 4)

    opt = get_option_str(option, item, config)
    try:
        return int(opt)
    except (TypeError, ValueError):
        return None


def get_option_float(option:str, item:str='DEFAULT', config:configparser=None):
    opt = get_option_str(option, item, config)
    try:
        return float(opt)
    except (TypeError, ValueError):
        return None


def get_option_bool(option:str, item:str='DEFAULT', config:configparser=None):
    opt = get_option_str(option, item, config)
    return False if opt == 'False' else bool(opt)


def get_option_list(option:str, item:str='DEFAULT', config:configparser=None):
    opt = get_option_str(option, item, config)
    l_opt = eval(opt)
    if isinstance(l_opt, list):
        return l_opt
    else:
        return None


def check_mimetype(filepath:str, mtype:str=None):
    try:
        file_mtype = mimetypes.guess_type(filepath)[0]
    except TypeError:
        return (False, 'unknown')

    if (mtype is not None) and (file_mtype is not None):
        return (True, file_mtype) if mtype in file_mtype else (False, file_mtype)
    else:
        return (False, file_mtype)


def check_acodecs(filepath:str, config:configparser=None):
    if filepath is None:
        return (False, None)
    
    ffprobe_path = get_option_str('ffprobe_path', 'path', config=config)
    command_list = [
        ffprobe_path, '-hide_banner', '-show_streams', '-show_entries', 'format=duration', '-print_format', 'json',
        '-select_streams', 'a', filepath
    ]
    # print(command_list)
    proc = subprocess.run(command_list, capture_output=True, text=True)
    j = json.loads(proc.stdout)
    audios = j.get('streams')

    if not audios:
        return (False, None)

    duration = j.get('format').get('duration')

    l = []
    for audio in audios:
        l.append({
            # 'track': audio.get('index'),
            'acodec': audio.get('codec_name'),
            'duration': duration
        })

    return (True, l)


def get_acodec(filepath:str=None, audios:list=None, track:int=None, config:configparser=None):
    if (filepath is None) and (audios is None):
        return None
    
    if audios is None:
        contains_audio, audios = check_acodecs(filepath, config)
        if not contains_audio:
            return None

    try:
        return audios[track].get('acodec')
    except IndexError:
        print(f"specified invalid track (#{track})")
        return None


def get_framerate(filepath:str, config:configparser=None):
    if filepath is None:
        return None
    ffprobe_path = get_option_str('ffprobe_path', 'path', config)
    command_list = [
        ffprobe_path, '-hide_banner', '-show_entries', 'stream=r_frame_rate', '-print_format', 'json',
        '-select_streams', 'v', filepath
    ]

    proc = subprocess.run(command_list, capture_output=True, text=True)
    j = json.loads(proc.stdout)

    try:
        r = eval(j.get('streams')[0].get("r_frame_rate"))
    except TypeError:
        return get_option_float('source_fps', 'video_params', config)

    # if not isinstance(r, float):
    #     return get_option_float('source_fps', 'video_params', config)
    return r


def set_file(filepath:str=None, config:configparser=None):
    """ Takes a filepath, validates the path, and returns the path with its audio information.

    Args:
        filepath (str, optional): An input filepath. If the file doesn't exist, an input dialog will open. Defaults to None.
        config (configparser, optional): 

    Returns:
        str: a filepath if the specified file is media or json; else returns None.
        list: a list from check_acodecs()[1] if the file is media contains audio; else returns None.
    """
    if (filepath is None) or not (os.path.isfile(filepath)):
        source_root_dir = get_option_str('source_rootdir', item='path', config=config)
        root = Tk().withdraw()
        print("Waiting for file selection ..")
        filepath = filedialog.askopenfilename(initialdir=source_root_dir)
    
    if not isinstance(filepath, str):
        print("File selection was cancelled.")
        sys.exit()

    contains_audio, audios = check_acodecs(filepath, config)

    if contains_audio:
        print("Received a media file.")
        return (filepath, audios)
    elif check_mimetype(filepath, "json")[0]:
        print("Received a json file.")
        return (filepath, None)
    else:
        print("Received an invalid file.")
        sys.exit()
        # return (None, None)

def set_markers(markersubdir:str='', config:configparser=None):
    """takes a directory, returns a list of tuple of (filepath, track).

    Args:
        markersubdir (str): a relative path to the directory from `markers_rootdir` where markers are stored
        config (configparser, optional):

    Returns:
        list: [(filepath, track), (filepath, track), ..]
    """
    markers_rootdir = get_option_str('markers_rootdir', 'path', config)
    markerdir = os.path.join(os.path.abspath(markers_rootdir), markersubdir)
    if not os.path.isdir(markerdir):
        print(f"{markerdir} is not a valid directory.")
        sys.exit()

    markers = []
    for f in os.listdir(markerdir):
        filepath = os.path.join(markerdir, f)
        contains_audio, audios = check_acodecs(filepath, config)
        if contains_audio:
            for track, _ in enumerate(audios):
                markers.append((filepath, track))

    if not markers:
        print(f"No markers found in {markerdir}")
        sys.exit()
    else:
        return markers


def set_source(source:str, track:int=None, config:configparser=None):
    """ verify the specified source file and track.

    Args:
        source (str): a filepath of the source
        track (int, optional): audio track to be processed. Defaults to None.
        config (configparser, optional):

    Returns:
        tuple: (filepath, track) if valid; else returns None.
    """
    if track is None:
        track = get_option_int('audio_track', 'params', config)
        print(f"Audio track is not specified. Using default (#{track}) ..")

    acodec = get_acodec(filepath=source, track=track, config=config)
    if acodec is not None:
        return (source, track)
    else:
        print(f"Couldn't find track #{track} in {source}")
        sys.exit()


def export_matches_json(d_matches:dict, export_rootdir:str=None, config:configparser=None):
    source_basename = get_filename_without_suffix(d_matches['source'])

    if export_rootdir is None:
        export_rootdir = get_option_str('export_rootdir', item='path', config=config)

    export_dir = os.path.join(export_rootdir, source_basename)
    os.makedirs(export_dir, exist_ok=True)

    export_basename = get_option_str('export_json_basename', 'path', config)
    target = os.path.join(export_dir, f'{export_basename}.json')

    with open(target, 'w') as f:
        json.dump(d_matches, f, indent=2, ensure_ascii=False)

    print(f'\nDumped matching results to {target}')
    return os.path.abspath(target)


def load_matches_json(jsonpath:str):
    mimetype_match, _ = check_mimetype(jsonpath, "json")
    if mimetype_match:
        with open(jsonpath) as f:
            d_matches = json.load(f)
        return d_matches
    else:
        return None

# def export_matches_ffmetadata(d_matches, export_dir_base=None, config=None):
#     source_path = d_matches['source']
#     source_basename = os.path.splitext(os.path.basename(source_path))[0]
#     if export_dir_base == None:
#         export_dir_base = get_option_str('export_dir_root', item='path', config=config)
#     export_dir = os.path.join(export_dir_base, source_basename)
#     os.makedirs(export_dir, exist_ok=True)
#     target = os.path.join(export_dir, 'matches.json')

#     sourcepath = d_matches['source']

#     source_path_split = os.path.splitext(source_path)
#     meta_path = source_path_split[0] + '.meta'

#     with open(meta_path, 'w') as f:
#         f.write(';FFMETADATA1\n')
#         f.write('title=' + config['output_title'] + '\n')
#         f.write('\n')
#         k = 0
#         last_time = 0
#         last_sample = 'N/A'
#         for match in matches:
#             end_time = int(round(match[1]))
#             if match[2] == True: # Not ignored
#                 f.write('#ignored=' + str(end_time * 1000) + ' (' + str(datetime.timedelta(seconds=end_time)) + ')\n')
#                 f.write('\n')
#             else:
#                 k += 1
#                 f.write('[CHAPTER]\n')
#                 f.write('TIMEBASE=1/1000\n')
#                 f.write('START=' + str(last_time * 1000) + '\n')
#                 f.write('END=' + str(end_time * 1000) + '\n')
#                 f.write('title=Chapter ' + str(k) + '\n')
#                 f.write('#human-start=' + str(datetime.timedelta(seconds=last_time)) + '\n')
#                 f.write('#human-end=' + str(datetime.timedelta(seconds=end_time)) + '\n')
#                 f.write('#sample=' + str(last_sample) + '\n')
#                 f.write('\n')
#                 last_time = end_time
#                 last_sample = samples[match[0]][2]
#         if last_time > 0:
#             k += 1
#             end_time = int(round((float(config['source_frame_end']) * hop_length) / sample_rate))
#             f.write('[CHAPTER]\n')
#             f.write('TIMEBASE=1/1000\n')
#             f.write('START=' + str(last_time * 1000) + '\n')
#             f.write('END=' + str(end_time * 1000) + '\n')
#             f.write('title=Chapter ' + str(k) + '\n')
#             f.write('#human-start=' + str(datetime.timedelta(seconds=last_time)) + '\n')
#             f.write('#human-end=' + str(datetime.timedelta(seconds=end_time)) + '\n')
#             f.write('#sample=' + str(last_sample) + '\n')
#             f.write('\n')
#     return None
