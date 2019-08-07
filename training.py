from utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir

class Trainer(object):
    def __init__(self, args):
        self.args = args
        if args.cfg_file is not None:
            cfg
        # Define Saver

        # Define Visdom

        # Define Dataloader

        # Define Network
        # initilize the network here.
        if args.net == 'vgg16':
            fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
        elif args.net == 'res101':
            fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
        # Define Optimizer

        # Define Criterion
        # Whether to use class balanced weights

        # Define Evaluater
        
        # Define lr scherduler
         
        # Resuming Checkpoint

        # Using cuda

        # Clear start epoch if fine-tuning

def main():
    from utils.hyp import parse_args
    args = parse_args()

    trainer = Trainer(args)
    # for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
    #     trainer.training(epoch)
    #     if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval -  1):
    #         trainer.validation(epoch)
    
    # trainer.writer.close()
     

if __name__ == "__main__":
    main()