import bisect
import os
import subprocess

from . import process_file as pf

def split_clips(
        d_matches:dict, config=None, keywords:list=None, export_dir:str=None, prefix:str=None, 
        separate_audio:bool=None, track:int=None):
    if isinstance(keywords, str):
        marker_keywords = [marker_keywords]
    
    source = d_matches['source']
    if track is None:
        track = d_matches['audio_track']

    if separate_audio is None:
        separate_audio = pf.get_option_bool('split_separate_audio', 'video_params', config)
    if keywords is None:
        keywords = pf.get_option_list('split_keywords', 'video_params', config)

        if (keywords is None) or (len(keywords) == 0):
            print("No keywords set for split_clips()")
            return -1

    target_markers = set()
    for keyword in keywords:
        target_markers |= set([k for k in list(d_matches.keys()) if keyword in k])

    if export_dir is None:
        export_rootdir = pf.get_option_str('export_rootdir', 'path', config)
        export_dir = os.path.join(export_rootdir, pf.get_filename_without_suffix(source))

    if prefix is None:
        prefix = pf.get_option_str('export_media_prefix', 'path', config)

    for marker in target_markers:
        t_start, t_end = '00:00', '00:00'
        for i, timestamp in enumerate(d_matches[marker]):
            t_end = str(timestamp)
            target_basename = prefix if prefix else pf.get_filename_without_suffix(marker)
            target_nosuffix = os.path.join(export_dir, f'{target_basename}_{i:03}')
            split_clip(source, target_nosuffix, start_time=t_start, end_time=t_end, 
                    config=config, separate_audio=separate_audio, track=track)
            t_start = str(timestamp)

        i+=1
        target_nosuffix = os.path.join(
                export_dir, f'{target_basename}_{i:03}')
        split_clip(source, target_nosuffix, start_time=t_start, 
                config=config, separate_audio=separate_audio, track=track)

    return i


def split_clip(
        source:str, target_nosuffix:str, start_time='00:00', end_time=None, 
        ffmpeg_path:str=None, config=None, loglevel:str=None, override:bool=None, 
        separate_audio:bool=None, track:int=None):
    if ffmpeg_path is None:
        ffmpeg_path = pf.get_option_str('ffmpeg_path', 'path', config)
    if loglevel is None:
        loglevel = pf.get_option_str('ffmpeg_loglevel', 'video_params', config)
    if separate_audio is None:
        separate_audio = pf.get_option_bool('split_separate_audio', 'video_params', config)
    if override is None:
        override = pf.get_option_bool('export_override', 'video_params', config)

    source_suffix = pf.get_file_suffix(source)
    acodec = pf.get_acodec(filepath=source, track=track, config=config)
    os.makedirs(os.path.dirname(target_nosuffix), exist_ok=True)

    command_list = [ffmpeg_path, '-loglevel', loglevel, '-stats',
                    '-ss', start_time]

    if end_time is not None:
        command_list.extend(['-to', end_time])

    command_list.extend(['-i', source])

    if separate_audio:
        target_v = ''.join([f"{target_nosuffix}_v", source_suffix])
        command_list.extend(['-c', 'copy', '-map', '0:v', target_v])

        target_a = '.'.join([f"{target_nosuffix}_a", acodec])
        if acodec == 'flac':
            command_list.extend(['-c', 'flac', '-map', f'0:a:{track}', target_a])
        else:
            command_list.extend(['-c', 'copy', '-map', f'0:a:{track}', target_a])
    else:
        target = ''.join([target_nosuffix, source_suffix])
        command_list.extend(['-c', 'copy', '-map', '0', target])

    if override:
        command_list.append('-y')

    # print(command_list)
    subprocess.run(command_list)

    return None


def generate_capture_tiles(
        d_matches:dict, config=None, keywords:list=None, prefix:str=None, 
        index_reference:str=None, index_offset:int=None, time_offset:float=None, export_dir:str=None):
    if isinstance(keywords, str):
        keywords = [keywords]

    source = d_matches['source']

    if keywords is None:
        keywords = pf.get_option_list('tiles_keywords', 'video_params', config)

        if (keywords is None) or (len(keywords) == 0):
            print("No keywords set for generate_capture_tiles()")
            return 1

    target_markers = set()
    for keyword in keywords:
        target_markers |= set([k for k in list(d_matches.keys()) if keyword in k])

    if export_dir is None:
        export_rootdir = pf.get_option_str('export_rootdir', 'path', config)
        export_dir = os.path.join(export_rootdir, pf.get_filename_without_suffix(source))
    if prefix is None:
        prefix = pf.get_option_str('export_media_prefix', 'path', config)
    if index_reference is None:
        index_reference = pf.get_option_str('tiles_index_reference', 'video_params', config)
    if index_offset is None:
        index_offset = pf.get_option_int('tiles_index_offset', 'video_params', config)
    if time_offset is None:
        time_offset = pf.get_option_float('tiles_start_offset', 'video_params', config)
    
    for marker in target_markers:
        target_basename = prefix if prefix else pf.get_filename_without_suffix(marker)

        for i, timestamp in enumerate(d_matches[marker], index_offset):
            if index_reference is not None:
                reference_marker = [k for k in list(d_matches.keys()) if index_reference in k]
                reference = d_matches[reference_marker[0]]
                idx = bisect.bisect_left(reference, timestamp)
                target = os.path.join(export_dir, f'{target_basename}_{idx:03}_{pf.get_filename_without_suffix(marker)}.png')
            else:
                target = os.path.join(export_dir, f'{os.path.splitext(marker)[0]}_{i:03}.png')
            time = str(timestamp + time_offset)
            generate_capture_tile(source, target, start_time=time, config=config)

    return i


def generate_capture_tile(
        source:str, target:str, config=None, start_time='00:00',
        ffmpeg_path=None, loglevel:str=None, override:bool=None,
        size:str=None, interval:int=None, scale:int=None, vframes:int=None, fps:float=None):
    if ffmpeg_path is None:
        ffmpeg_path = pf.get_option_str('ffmpeg_path', 'path', config)
    if loglevel is None:
        loglevel = pf.get_option_str('ffmpeg_loglevel', 'video_params', config)
    if override is None:
        override = pf.get_option_bool('export_override', 'video_params', config)
    
    if size is None:
        size = pf.get_option_str('tiles_size', 'video_params', config)
    if interval is None:
        interval = pf.get_option_float('tiles_interval', 'video_params', config)
    if scale is None:
        scale = pf.get_option_int('tiles_scale', 'video_params', config)
    if vframes is None:
        vframes = pf.get_option_int('tiles_vframes', 'video_params', config)
    if fps is None:
        fps = pf.get_framerate(source, config)


    os.makedirs(os.path.dirname(target), exist_ok=True)

    command_list = [ffmpeg_path, '-an', '-loglevel', loglevel, '-stats',
                    '-ss', start_time,
                    '-i', source,
                    '-vf', f'select=not(mod(n\,{fps*interval})),scale={scale}:-1,tile={size}',
                    '-vframes', str(vframes),
                    target]
    if override:
        command_list.append('-y')

    print(f'Generate {target}')
    # print(' '.join(command_list))
    subprocess.run(command_list, check=False)

    return target


def fix_hevc_framerate(
        source:str, export_dir:str=None, ffmpeg_path:str=None, config=None, 
        loglevel:str=None, fps:float=None, thread_queue_size=8192, override:bool=None):

    sourcepath_split = source.split(".")
    source_remuxed = f'{("").join(sourcepath_split[:-1])}_remuxed.{sourcepath_split[-1]}'

    if export_dir is None:
        export_dir_root = pf.get_option_str('export_rootdir', 'path', config)
        export_dir = os.path.join(export_dir_root, pf.get_filename_without_suffix(source_remuxed))

    os.makedirs(export_dir, exist_ok=True)
    source_remuxed = f'{export_dir}/{source_remuxed.split("/")[-1]}'

    if ffmpeg_path is None:
        ffmpeg_path = pf.get_option_str('ffmpeg_path', 'path', config)
    if loglevel is None:
        loglevel = pf.get_option_str('ffmpeg_loglevel', 'video_params', config)
    if fps is None:
        fps = pf.get_framerate(source, config)
    if override is None:
        override = pf.get_option_bool('export_override', 'video_params', config)

    command_list = [ffmpeg_path, '-an', '-loglevel', loglevel, '-stats',
                    '-i', source, '-bsf:v', 'hevc_metadata', '-c:v', 'copy', '-f', 'hevc', '-', '|', 
                    ffmpeg_path, '-loglevel', loglevel, '-fflags', '+genpts', '-r', f'{fps}', '-f', 'hevc', '-thread_queue_size', f'{thread_queue_size}',
                    '-i', '-', '-vn', '-i', source,
                    '-map', '0:v', '-c:v', 'copy',
                    '-map', '1:a', '-c:a', 'copy',
                    source_remuxed]
    if override:
        command_list.append('-y')

    print(f'Remuxing source to {source_remuxed} ..')
    if os.path.exists(source_remuxed):
        print('remuxed file already exists')
        return os.path.abspath(source_remuxed)
    else:
        cmd = (' ').join(command_list)
        subprocess.run(cmd, shell=True)
        return os.path.abspath(source_remuxed)
