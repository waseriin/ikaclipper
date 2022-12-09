import argparse

from ikaclipper import pa, pf, pv

def main():
    parser = argparse.ArgumentParser()

    config = pf.load_config()

    audio_track = pf.get_option_int('audio_track', 'params', config)
    markerpath = pf.get_option_str('markers_rootdir', 'path', config)

    split_clips = pf.get_option_bool('split_clips', 'params', config)
    split_keywords = pf.get_option_list('split_keywords', 'video_params', config)
    separate_audio = pf.get_option_bool('split_separate_audio', 'video_params', config)

    generate_tiles = pf.get_option_bool('generate_tiles', 'params', config)
    tiles_keywords = pf.get_option_list('tiles_keywords', 'video_params', config)
    tiles_reference = pf.get_option_str('tiles_index_reference', 'video_params', config)

    prefix = pf.get_option_str('export_media_prefix', 'path', config)


    parser.add_argument(
        "-i", "--input", dest="filepath",
        help="the source file path (accepts video/audio or json)")
    parser.add_argument(
        "-a", "--audio-track", type=int, default=audio_track, dest="track",
        help=f"the audio track in video to process (default: #{audio_track})")
    parser.add_argument(
        "-m", "--marker", default='',
        help=f"the marker subdirectory follows {markerpath}")

    parser.add_argument(
        "-s", "--split", action="store_true", default=split_clips,
        help="split clips regardless of config")
    parser.add_argument(
        "-sk", "--split-keywords", nargs='*', default=split_keywords, dest="sk",
        help="overrides keywords to split source; accepts multiple keywords with input like `-sk SK1 SK2`")
    parser.add_argument(
        "-sa", "--split-audio", action="store_true", default=separate_audio, dest="sa",
        help="separate audio and video files when split clips")

    parser.add_argument(
        "-t", "--tiles", action="store_true", default=generate_tiles, 
        help="generate tiles regardless of config")
    parser.add_argument(
        "-tk", "--tiles-keywords", nargs='*', default=tiles_keywords, dest="tk",
        help="overrides keywords to generate tiles; accepts multiple keywords with input like `-tk TK1 TK2`")
    parser.add_argument(
        "-tr", "--tiles-reference", default=tiles_reference, dest="tr",
        help="overrides index reference marker for tile generation")

    parser.add_argument(
        "-p", "--prefix", default=prefix,
        help="overrides filename prefix used for file generation")

    args = vars(parser.parse_args())
    # print(args)

    filepath, audios = pf.set_file(args['filepath'], config)

    if audios:
        jsonpath = _process_media(filepath, config, args)
    elif filepath is None:
        raise(IOError)
    else:
        jsonpath = filepath

    _process_json(jsonpath, config, args)


def _process_media(mediapath:str, config, d:dict):
    markers = pf.set_markers(d['marker'], config)
    t_source = pf.set_source(mediapath, d['track'], config)
    d_matches = pa.main(t_source, markers, config=config)
    return pf.export_matches_json(d_matches, config=config)


def _process_json(jsonpath:str, config, d:dict):
    d_matches = pf.load_matches_json(jsonpath)
    if d['tiles']:
        print('tiles will be generated')
        pv.generate_capture_tiles(d_matches, config, keywords=d['tk'], prefix=d['prefix'], index_reference=d['tr'])
    if d['split']:
        print('files will be split')
        pv.split_clips(d_matches, config, d['sk'], prefix=d['prefix'], separate_audio=d['sa'], track=d['track'])
    return None


if __name__ == '__main__':
    main()
