# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from alpr.detector import PlateDetector
import cv2
from timeit import default_timer as timer
from argparse import ArgumentParser
import os
from pathlib import Path


def main_demo(args):
    input_size = args.input_size
    video_path = args.file_source
    weights_path = f'./alpr/models/detection/tf-yolo_tiny_v4-{input_size}x{input_size}-custom-anchors/'
    iou = 0.45
    score = 0.25
    # Detector
    detector_patente = PlateDetector(
        weights_path, input_size=input_size, iou=iou, score=score)
    print("Video from: ", video_path)
    vid = cv2.VideoCapture(video_path)

    frame_id = 0
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            break
        # Preprocess frame
        input_img = detector_patente.preprocess(frame)
        # Inference
        yolo_out = detector_patente.predict(input_img)
        print('FRAME:', frame_id)
        # Bounding Boxes despues de NMS
        bboxes = detector_patente.procesar_salida_yolo(yolo_out)
        # Mostrar predicciones
        start = timer()
        frame_w_preds = detector_patente.draw_bboxes(frame, bboxes)
        end = timer()
        # Tiempo de inferencia
        exec_time = end - start
        fps = 1. / exec_time
        if args.mostrar_benchmark and args.mostrar_resultados:
            display_bench = f'ms: {exec_time:.4f} FPS: {fps:.0f}'
            fontScale = 1.5
            cv2.putText(frame_w_preds, display_bench, (5, 45), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (10, 140, 10), 4)
        elif args.mostrar_benchmark:
            print(f'Inferencia\tms: {exec_time:.5f}\t', end='')
            print(f'FPS: {fps:.0f}')
        if args.mostrar_resultados:
            result = cv2.cvtColor(frame_w_preds, cv2.COLOR_RGB2BGR)
            print('RESULTADO', result)
            # Show results
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("result", result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_id += 1

def main_img(args):
    input_size = args.input_size
    file_path = args.file_source
    weights_path = f'./alpr/models/detection/tf-yolo_tiny_v4-{input_size}x{input_size}-custom-anchors/'
    iou = 0.45
    score = 0.25
    # Detector
    detector_patente = PlateDetector(
        weights_path, input_size=input_size, iou=iou, score=score)
    dir_list = [file_path]
    basefolder, _ = os.path.split(file_path)
    if args.is_folder:
        dir_list = os.listdir(file_path)
        try:
            os.mkdir(Path(basefolder) / 'cropped')
        except FileExistsError:
            pass
    im_cnt = 1
    for file_path in dir_list:
        print(im_cnt, "Image from: ", file_path)
        im_cnt+=1
        frame = cv2.imread(Path(basefolder) / file_path)
        if frame is None:
            continue
        frame_height, frame_width, _ = frame.shape
        print('Size:', frame.shape) 
        basename = os.path.basename(file_path)
        base_no_ext, ext = os.path.splitext(basename)
        newname = f'{base_no_ext}.cropped{ext}'
        # Preprocess frame
        input_img = detector_patente.preprocess(frame)
        # Inference
        yolo_out = detector_patente.predict(input_img)
        # Bounding Boxes despues de NMS
        bboxes = detector_patente.procesar_salida_yolo(yolo_out)
        # Mostrar predicciones
        start = timer()
        possible_plates = []
        for x1, y1, x2, y2, score in detector_patente.yield_coords(frame, bboxes):
            possible_plates.append({'plate': frame[y1:y2, x1:x2], 'coords': [(x1,y1), (x2,y2)], 'score': score})
        end = timer()
        # Tiempo de inferencia
        exec_time = end - start
        fps = 1. / exec_time
        #result = cv2.cvtColor(frame_w_preds, cv2.COLOR_RGB2BGR)
        #print('RESULTADO', result)
        # Show results

        if args.mostrar_resultados:
            cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        max_plate = None
        max_score = 0
        max_size = 0
        for i, possible_plate in enumerate(possible_plates):
            print(f'PLATE #{i}, score:', possible_plate['score'], 'coords:', possible_plate['coords'], 'shape:', possible_plate['plate'].shape) 
            height, width, _ = possible_plate['plate'].shape
            rel_size = height*width/(frame_height/frame_width)
            if 1 < width/height < 3 and ((rel_size > 8*max_size and possible_plate['score'] > 0.75) or possible_plate['score'] > max_score):
                max_plate = possible_plate['plate']
                max_score = possible_plate['score']
                max_size = rel_size
            if args.mostrar_resultados:
                cv2.imshow("result", possible_plate['plate'])
                if cv2.waitKey(3000) & 0xFF == ord('q'):
                    return
        if max_plate is not None:
            cv2.imwrite(Path(basefolder) / 'cropped' / newname, max_plate)
        else:
            print('NO PLATE FOUND')




if __name__ == '__main__':
    try:
        parser = ArgumentParser()
        parser.add_argument("-f", "--fuente-video", dest="file_source",
                            required=True, type=str, help="Archivo de entrada, para video: 0,\
                                camara ip: rtsp://user:pass@IP:Puerto, video en disco: C:/.../vid.mp4")
        parser.add_argument("-g", "--imagen", dest="is_image",
                            action='store_true', help="Archivo de entrada es imagen")
        parser.add_argument('-c', '--carpeta', dest='is_folder',
                            action='store_true', help='Ruta de entrada es carpeta (procesar imágenes dentro)')
        parser.add_argument("-i", "--input-size", dest="input_size",
                            default=512, type=int, help="Modelo a usar, opciones: 384, 512, 608")
        parser.add_argument("-m", "--mostrar-resultados", dest="mostrar_resultados",
                            action='store_true', help="Mostrar los frames con las patentes dibujadas")
        parser.add_argument("-b", "--benchmark", dest="mostrar_benchmark",
                            action='store_true', help="Mostrar tiempo de inferencia (ms y FPS)")
        args = parser.parse_args()
        if args.is_image or args.is_folder:
            main_img(args)
        else:
            main_demo(args)
    except Exception as e:
        print(e)
