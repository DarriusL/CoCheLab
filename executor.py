import argparse, sys, logging
from data.generator import run_pcr
from lib import glb_var
from lib.callback import Logger, CustomException
from Room.work import run_work

if __name__ == '__main__':
    glb_var.__init__();
    log = Logger(
        level = logging.DEBUG,
        filename = './cache/logger/logger.log',
    ).get_log()
    glb_var.set_value('logger', log);
    parse = argparse.ArgumentParser();
    parse.add_argument('--data_process', '-dp', type = str, default = None, help = 'type for data process(None/lite/complete)');
    parse.add_argument('--config', '-cfg', type = str, default = None, help = 'config for run');
    parse.add_argument('--saved_config', '-sc', type = str, default = None, help = 'path for saved config to test')
    parse.add_argument('--mode', type = str, default = 'train', help = 'train/test/train_and_test')

    args = parse.parse_args();

    #execute date process command
    if args.data_process is not None:
        dp_config = {
            'type': args.data_process,
            'tgt': None
        }
        run_pcr(cfg = dp_config);
        sys.exit(0);

    #execute work command
    if args.config is not None:
        if args.mode not in ['train', 'train_and_test']:
            log.error('The using of the configuration file does not support the [test] mode. \n'
                      'Prompt: first use the [train] mode and then use the command [-- saved_config] to run the [test] mode, \n'
                      'or directly use the [train_and_test] mode');
            raise CustomException('ModeError');
        run_work(args.config, args.mode);
    elif args.saved_config is not None:
        if args.mode in ['train', 'train_and_test']:
            log.error('The saved model file only supports test mode');
            raise CustomException('ModeError');
        run_work(args.saved_config, args.mode);

    
    
        
    