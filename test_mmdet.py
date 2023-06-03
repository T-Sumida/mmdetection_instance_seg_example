import mmcv
import mmdet
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.registry import VISUALIZERS


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection instance segmentation demo')
    parser.add_argument('image', help='image file')
    parser.add_argument('--out', type=str, default=None, help='Output image file')
    parser.add_argument("--config", type=str, required=True, help="config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint file")
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()

    register_all_modules()

    model = init_detector(args.config, args.checkpoint,
                          device='cuda:0')  # or device='cuda:0'

    # Use the detector to do inference
    image = mmcv.imread(
        args.image, channel_order='rgb')
    result = inference_detector(model, image)

    visualizer = VISUALIZERS.build(model.cfg.visualizer)

    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_detector
    visualizer.dataset_meta = model.dataset_meta
    # show the results
    visualizer.add_datasample(
        'result',
        image,
        data_sample=result,
        draw_gt=None,
        wait_time=0,
    )
    visualizer.show()

    if args.out is not None:
        frame = visualizer.get_image()
        frame = mmcv.imconvert(frame, 'rgb', 'bgr')
        mmcv.imwrite(frame, args.out)


if __name__ == "__main__":
    main()
