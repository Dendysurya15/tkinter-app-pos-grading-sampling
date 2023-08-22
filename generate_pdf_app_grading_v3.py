# limit the number of cpus used by high performance libraries
import os
from urllib import response
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import aiohttp
import asyncio

from reportlab.pdfgen import canvas
from reportlab.lib import colors as colorPdf

from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Image as ImgRl
from reportlab.platypus import Table, TableStyle, Paragraph
from reportlab.platypus import Spacer
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter

import sys
sys.path.insert(0, './yolov5')

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import time
import datetime
from datetime import datetime, timedelta
import requests
import pytz
import PIL
from PIL import Image

from models.experimental import attempt_load
from utils.downloads import attempt_download
from models.common import DetectMultiBackend
from utils.datasets import LoadImages, LoadStreams
from utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, 
                                  check_imshow, xyxy2xywh, increment_path)
from utils.torch_utils import select_device, time_sync
from utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
count = 0
count_now = 0
data = []
roi = 0.0

yolo_model_str = ''
confidence = 0.0
iou = 0.0
unripe = 0
ripe = 0
overripe = 0
empty_bunch = 0
abnormal = 0
kastrasi = 0
long_stalk = 0
countUnripe = 0
countRipe = 0
countOverripe = 0
countEmptybunch = 0
countAbnormal = 0
countLongStalk = 0
countKastrasi = 0
countLongStalk = 0
prctgUR = 0
prctgRP = 0
prctgOV = 0
prctgEB = 0
prctgAB = 0
prctgKS = 0
prctgLS = 0
baseSkorUr = 1
baseSkorRp = 3
baseSkorOv = 2
baseSkorEm = 1
baseSkorAb = 2
baseSkorKas = 2
id_ffb =  []
time_ffb = []
last_mins = 0
mins = 0.25
bad_ffb = False
good_ffb = False
best_ffb = False
worst_ffb = False
maxArea = 40000
timer = 25
url = 'https://srs-ssms.com/post-py.php'
headers = {"content-type": "application/x-www-form-urlencoded",
          'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.80 Safari/537.36'}
#set the timezone
tzInfo = pytz.timezone('Asia/Bangkok')
current_date = datetime.now()
formatted_date = current_date.strftime('%Y-%m-%d')

dateStart = datetime.now(tz=tzInfo).strftime("%Y-%m-%d %H:%M:%S")

def generate_report(content, path):
    
    arrData = content.split(';')

    # print(arrData)

    TotalJjg = 0
    prctgUnripe = 0
    prctgRipe = 0
    prctgEmptyBunch = 0
    prctgOverripe = 0
    prctgAbnormal = 0
    prctgKastrasi = 0
    prctgLongStalk = 0
    TotalRipeness = 0
    no_tiket = str(arrData[0])
    no_plat = str(arrData[1])
    nama_driver = str(arrData[2])
    bisnis_unit = str(arrData[3]).replace('\n','')
    divisi = str(arrData[4])
    blok = str(arrData[5])
    status = str(arrData[6])    
    str(unripe) + ";" + str(ripe)+ ";" + str(overripe) + ";" + str(empty_bunch) + ";" + str(abnormal)
    Ripe = arrData[9]
    Overripe = arrData[10]
    Unripe = arrData[8]
    EmptyBunch = arrData[11]
    Abnormal = arrData[12]
    Kastrasi = arrData[13]
    LongStalk = arrData[14]
    dateStart = str(arrData[15])
    dateEnd = str(arrData[16]).replace('\n','')
    
    TotalJjg = int(Ripe) + int(Overripe) + int(Unripe) + int(EmptyBunch) + int(Abnormal) + int(Kastrasi)
    if int(TotalJjg) != 0:

        prctgUnripe = round((int(Unripe) / int(TotalJjg)) * 100,2)
        prctgRipe = round((int(Ripe) / int(TotalJjg)) * 100,2)
        prctgEmptyBunch = round((int(EmptyBunch) / int(TotalJjg)) * 100,2)
        prctgOverripe = round((int(Overripe) / int(TotalJjg)) * 100,2)
        prctgAbnormal = round((int(Abnormal) / int(TotalJjg)) * 100,2)
        prctgKastrasi = round((int(Kastrasi) / int(TotalJjg)) * 100,2)
        prctgLongStalk = round((int(LongStalk) / int(TotalJjg)) * 100,2)
        
        TotalRipeness = round((int(Ripe) / int(TotalJjg)) * 100,2)

    date = dateStart.split(' ')
    
    
    TabelAtas = [
        ['No Tiket',   str(no_tiket),'','','', 'Waktu Mulai',  str(dateStart)],
        ['Bisnis Unit',  str(bisnis_unit),'','','','Waktu Selesai', str(dateEnd)],
        ['Divisi',   str(divisi),'','','','No. Plat',str(no_plat)],
        ['Blok',  str(blok),'','','','Driver',str(nama_driver)],
        ['Status',  str(status)]
    ]

    colEachTable1 = [1.2*inch, 1.6*inch,  0.8*inch, 0.8*inch, 0.8*inch, 1.2*inch, 1.6*inch]

    TabelBawah = [
        ['Total\nJanjang', 'Ripe', 'Overripe', 'Unripe', 'Empty\nBunch','Abnormal','Kastrasi','Tangkai\nPanjang', 'Total\nRipeness'],
        [TotalJjg, Ripe + ' (' +  str(prctgRipe) + ' %)', Overripe + ' (' +  str(prctgOverripe)+ ' %)', Unripe + ' (' + str(prctgUnripe) +' %)',EmptyBunch + ' (' + str(prctgEmptyBunch) +  ' %)', Abnormal + ' (' + str(prctgAbnormal)+ ' %)', Kastrasi + ' (' +  str(prctgKastrasi)+ ' %)',LongStalk + ' (' +  str(prctgLongStalk)+ ' %)', str(TotalRipeness) + ' % ']
    ]   


    colEachTable2 = [0.9*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.9*inch]

    spacer = Spacer(1, 0.25*inch)



    # Create the styles for the tables
    # styles = getSampleStyleSheet()
    # black = colorPdf.black
    # style_table = TableStyle([
    #     ('BACKGROUND', (0, 0), (-1, 0), colorPdf.gray),
    #     # ('TEXTCOLOR', (0, 0), (-1, 0), colorPdf.whitesmoke),
    #     ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
    #     ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    #     # ('FONTSIZE', (0, 0), (-1, 0), 14),
    #     ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    #     # ('BACKGROUND', (0, 1), (-1, -1), colorPdf.beige),
    #     # ('TEXTCOLOR', (0, 1), (-1, -1), colorPdf.black),
    #     ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
    #     # ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
    #     # ('FONTSIZE', (0, 1), (-1, -1), 12),
    #     # ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
    #     # ('SPAN',(3,0),(5,5)),
    #     ('GRID', (0, 0), (2, 4), 1, colorPdf.black),
    #     ('GRID', (6, 0), (9, 9), 1, colorPdf.black)
    #  ])


    
    # imgBest = str(path) + '/' + str(date[0])  + '_best_.JPG'
    # LOGGER.info(imgPath)
    
    dateImage = date[0]
    checkImgBest = os.path.isfile(str(path) + '/' + '_best_.JPG')
    if(checkImgBest):
        image = ImgRl("img_inference/" + '_best_.JPG')
    else:
        image = ImgRl("img_inference/no_image.png")
        
    checkImgWorst = os.path.isfile(str(path) + '/' + '_worst_.JPG')
    if(checkImgWorst):
        image2 = ImgRl("img_inference/" + '_worst_.JPG')
    else:
        image2 = ImgRl("img_inference/no_image.png")
        
    # LOGGER.info(checkImgBest)
    # LOGGER.info(checkImgWorst)
    # image = ImgRl(path + str(date[0])  + '_best_.JPG')
    # image2 = ImgRl("img_inference/" + str(date[0]) + '_worst_.JPG')
    logoCbi = ImgRl("Logo CBI.png")
    max_width = 285  # The maximum allowed width of the image
    max_widthLogo = 70  # The maximum allowed width of the image
    widthLogo = min(logoCbi.drawWidth, max_widthLogo)  # The desired width of the image
    width1 = min(image.drawWidth, max_width)  # The desired width of the image
    width2 = min(image2.drawWidth, max_width)  # The desired width of the image
    image._restrictSize(width1, image.drawHeight)
    image2._restrictSize(width2, image2.drawHeight)
    logoCbi._restrictSize(widthLogo, logoCbi.drawHeight)

    styleTitle = ParagraphStyle(name='Normal', fontName='Helvetica-Bold',fontSize=12,fontWeight='bold')
    t1 = Paragraph('CROP RIPENESS CHECK REPORT', style=styleTitle)
    # title = [[logoCbi, t1]]
    # titleImg = Table(title, [1.3*inch,6.7*inch])
    # titleImg.setStyle(TableStyle([
    #     ('GRID', (0, 0), (-1, -1), 1, colorPdf.black), 
    #    ('ALIGN', (0, 0), (-1, -1), 'CENTER')
    # ]))

    title = [[logoCbi, 'CROP RIPENESS CHECK REPORT']]
    titleImg = Table(title, [1.3*inch,6.7*inch])
    titleImg.setStyle(TableStyle([
       ('GRID', (0, 0), (-1, -1), 1, colorPdf.black), 
       ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
       ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 15),
    ]))
    
    dataImg = [[image, image2],['Kondisi Paling Baik', 'Kondisi Paling Buruk']]
    tblImg = Table(dataImg, [4.0*inch,4.0*inch])
    tblImg.setStyle(TableStyle([
    #    ('GRID', (0, 0), (-1, -1), 1, colorPdf.black), 
       ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
       ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))

    
    # p1 = Paragraph('KONDISI TBS :')
    dataP1 = [['KONDISI TBS : ']]
    tblP1 = Table(dataP1,[8*inch])
    tblP1.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
    ]))


    doc = SimpleDocTemplate("pdf/" + no_tiket + '_' +bisnis_unit +'_' + divisi + '_' + no_plat + '.pdf', pagesize=letter)
    
    table1 = Table(TabelAtas,colWidths=colEachTable1)
    table1.setStyle(TableStyle([
        ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
        ('GRID', (0, 0), (1, 4), 1, colorPdf.black),
        ('GRID', (5, 0), (8, 3), 1, colorPdf.black)
    ]))
    table2 = Table(TabelBawah, colWidths=colEachTable2)
    table2.setStyle(TableStyle([
        ('ALIGN', (0, 0), (8, 0), 'CENTER'),
        ('ALIGN', (0, 1), (8, 1), 'LEFT'),
        ('VALIGN', (0, 0), (8, 0), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 1, colorPdf.black)
    ]))



    elements = []
    elements.append(titleImg)
    elements.append(spacer)
    elements.append(table1)
    elements.append(spacer)
    elements.append(tblP1)
    # elements.append(Spacer(1,0.1*inch))
    elements.append(tblImg)
    elements.append(spacer)
    elements.append(table2)
    doc.build(elements)
    # # Build the PDF document
    # doc.build(elements)


    # # Create a canvas object
    # c = canvas.Canvas("hello-world.pdf", pagesize='A4')

    # # Add text to the canvas
    # c.drawString(100, 750, str(arrData[0]))

    # # Save the PDF file
    # c.save()

def detect(opt):
    global bad_ffb, good_ffb, last_mins, best_ffb, worst_ffb, formatted_date
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, project, name, exist_ok= \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.name, opt.exist_ok
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    device = select_device(opt.device)
    
    lastDate = datetime.now(tz=tzInfo)+timedelta(seconds=timer, minutes=0, hours=0)

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(deep_sort_model,
                        device,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        )

    # Initialize
    
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Directories
    log_dir = Path(os.getcwd() + '/log')
    log_kastrasi = Path(os.getcwd() + '/log_kastrasi')
    log_kastrasi.mkdir(parents=True, exist_ok=True)  # make dir
    log_dir.mkdir(parents=True, exist_ok=True)  # make dir
    log_inference = Path(os.getcwd() + '/log_inference_sampling')
    log_inference.mkdir(parents=True, exist_ok=True)  # make dir
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0

    skor_tertinggi = 0
    skor_terendah = 100
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
        pred = model(img, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # Process detections
        
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            s += '%gx%g ' % img.shape[2:]  # print string
            annotator = Annotator(im0, line_width=3, pil=not ascii)
            w, h = im0.shape[1], im0.shape[0]
            badCount = 0
            goodCount = 0
            bestCount = 0
            worstCount = 0
            
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                skorTotal = 0
                countOnFrame = 0
                nilai = 0
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    countOnFrame += int(n)
                    # buahApa += f"{n} {names[int(c)]}, "
                    # arrTes.append(int(c))
                    if names[int(c)] == "unripe":
                        skorTotal += int(n) * baseSkorUr
                    elif names[int(c)] == "ripe":
                        skorTotal += int(n) * baseSkorRp
                    elif names[int(c)] == "overripe":
                        skorTotal += int(n) * baseSkorOv
                    elif names[int(c)] == "empty_bunch":
                        skorTotal += int(n) * baseSkorEm
                    elif names[int(c)] == "abnormal":
                        skorTotal += int(n) * baseSkorAb
                    elif names[int(c)] == "kastrasi":
                        skorTotal += int(n) * baseSkorKas
                    else:
                        skorTotal += 0
                
                # LOGGER.info('Total buah satu frame: ' + str(countOnFrame))
                # LOGGER.info(buahApa)
                # LOGGER.info('skort total ' + str(countOnFrame) + ' buah : '  + str(skorTotal))
                nilai = (skorTotal * 100 ) / (countOnFrame * 3)


                # LOGGER.info('nilai skor_tertinggi saat ini : ' + str(skor_tertinggi))
                # LOGGER.info('nilai skor_terendah saat ini : '  + str(skor_terendah))
                # LOGGER.info('nilai : ' + str(round(nilai,2)))
                
                if nilai > skor_tertinggi:
                    skor_tertinggi = round(nilai,2)
                    save_img_inference_sampling(im0, str(Path(os.getcwd())), '/img_inference/_worst_')
                    LOGGER.info('tersimpan best')
                if nilai < skor_terendah:
                    skor_terendah = round(nilai,2)
                    save_img_inference_sampling(im0, str(Path(os.getcwd())),  '/img_inference/_best_')
                    LOGGER.info('terburuk best')
                # LOGGER.info('--')
                # LOGGER.info('nilai skor_tertinggi : ' + str(skor_tertinggi))
                # LOGGER.info('nilai skor_terendah : '  + str(skor_terendah))

                    # if names[int(c)] == "unripe" or names[int(c)] == "abnormal" or names[int(c)] == "empty_bunch":
                    #     badCount += int(n)
                    # elif names[int(c)] == "ripe":
                    #     goodCount += int(n)
        
                
                # LOGGER.info(skor_terendah)
                # LOGGER.info(skor_tertinggi)


                if countOnFrame <= 5:
                    LOGGER.info('Kurang dari 5')
                else:
                    LOGGER.info('--')
                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                t4 = time_sync()
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)):



                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        #count
                        # LOGGER.info('cls = '  + str(cls))
                        c = int(cls)  # integer class 
                        
                        # wideArea = (bboxes[2] - bboxes[0] ) * (bboxes[3] - bboxes[1])
                        try:
                            bboX = 1
                            bboY = 1
                            if(bboxes[2]>bboxes[0]):
                                bboX = bboxes[2]-bboxes[0]
                            elif(bboxes[2]==bboxes[0]):
                                bboX = 1
                            else:
                                bboX = bboxes[0]-bboxes[2]

                            if(bboxes[3]>bboxes[1]):
                                bboY = bboxes[3]-bboxes[1]
                            elif(bboxes[3]==bboxes[1]):
                                bboY = 1
                            else:
                                bboY = bboxes[1]-bboxes[3]

                            wideArea = bboX * bboY

                            wideArea 
                            pinggir = False
                            if bboxes[0] == 0 or bboxes[1] == 0 or bboxes[2] == 0 or bboxes[3] == 0:
                                pinggir = True
                            if pinggir:
                                wideArea = wideArea + (wideArea * 20 / 100)

                        except:
                            LOGGER.info('Terjadi kesalahan terkait Wide Box')
                            wideArea = maxArea - 1
                        label = f'{id} {names[c]} {conf:.2f} {wideArea:,}' #clue1
                        # label2 = f'{id} '+ 'Kastrasi' + f' {wideArea:,}' #clue1
                        # print("Luas buah " + str(wideArea))
                        if wideArea < maxArea and c != 5:
                            # print('kastrasi')
                            count_obj(bboxes, w, h, id, 6, log_inference)
                            annotator.box_label(bboxes, 'Kastrasi', color=colors(5,True))
                        else:
                            if c == 5:
                                count_obj(bboxes, w, h, id, c, log_inference)
                                annotator.box_label(bboxes, label, color=colors(6, True))
                            else:
                                count_obj(bboxes, w, h, id, c, log_inference)
                                annotator.box_label(bboxes, label, color=colors(c, True))
                     

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                           
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))

                # LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

            else:
                deepsort.increment_ages()
                # LOGGER.info('No detections')
                
            
            # Stream results
            im0 = annotator.result()
            
            
            
            if show_vid:
                global count, count_now, unripe, ripe, overripe, empty_bunch, abnormal, long_stalk, kastrasi, countUnripe, countRipe, countOverripe, countEmptybunch, countAbnormal, countLongStalk, countKastrasi, prctgUR, prctgRP, prctgOV, prctgEB, prctgAB, prctgKS, prctgLS
                color= (0, 255, 0)
                start_point = (0, int(h*(1-roi)))
                end_point = (w, int(h*(1-roi)))
                cv2.line(im0, start_point, end_point, color, thickness=2)
                thickness = 3 
                org= (150, 150)
                font = 2
                fontScale = 3
                fontRipeness = 1
                cv2.putText(im0, str(datetime.now(tz=tzInfo).strftime("%A,%d-%m-%Y %H:%M:%S")), org, font, 1.5, color, 2, cv2.LINE_AA)
                cv2.putText(im0, str(count), (15,260), font, fontRipeness, (255, 255, 255), thickness, cv2.LINE_AA)
                # cv2.putText(im0, str(count_now), (150,300), font, fontRipeness, (255, 0, 255), thickness, cv2.LINE_AA)
                cv2.putText(im0, "unripe: " + str(unripe) + " / " + str(prctgUR) + " %", (15, 350), font, fontRipeness, colors(0, True), 2, cv2.LINE_AA)
                cv2.putText(im0, "ripe: " + str(ripe)+ " / " + str(prctgRP) + " %", (15, 400), font, fontRipeness, colors(1, True), 2, cv2.LINE_AA)
                cv2.putText(im0, "overripe: " + str(overripe)+ " / " + str(prctgOV) + " %", (15, 450), font, fontRipeness, colors(2, True), 2, cv2.LINE_AA)
                cv2.putText(im0, "empty bunch: " + str(empty_bunch)+ " / " + str(prctgEB) + " %", (15, 500), font, fontRipeness, colors(3, True), 2, cv2.LINE_AA)
                cv2.putText(im0, "abnormal: " + str(abnormal)+ " / " + str(prctgAB) + " %", (15, 550), font, fontRipeness, colors(4, True), 2, cv2.LINE_AA)
                cv2.putText(im0, "kastrasi: " + str(kastrasi) + " / " + str(prctgKS)  + " %", (15, 600), font, fontRipeness, colors(5, True), 2, cv2.LINE_AA)
                cv2.putText(im0, "tangkai panjang: " + str(long_stalk) + " / " + str(prctgLS)  + " %", (15, 650), font, fontRipeness, colors(6, True), 2, cv2.LINE_AA)
                # cv2.putText(im0, "long stalk: " + str(long_stalk) + " / " + str(countLongStalk) , (15, 650), font, fontRipeness, colors(6, True), 2, cv2.LINE_AA)
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    
                    save_inference_data(str(unripe) + ";" + str(ripe)+ ";" + str(overripe) + ";" + str(empty_bunch) + ";" + str(abnormal) + ";" + str(kastrasi) + ";" + str(long_stalk) +';' + str(dateStart) + ";" + str(datetime.now(tz=tzInfo).strftime("%Y-%m-%d %H:%M:%S")), str(log_inference), )
                    append_inference_data(";"+str(unripe) + ";" + str(ripe)+ ";" + str(overripe) + ";" + str(empty_bunch) + ";" + str(abnormal) + ";" + str(kastrasi) + ";" + str(long_stalk) + ";" + str(dateStart) + ";" + str(datetime.now(tz=tzInfo).strftime("%Y-%m-%d %H:%M:%S")), str(log_inference), no_line_log)
                    delete_inference_file(str(log_inference) + '/' + path_plat + '.TXT')

                # def export_report():
                    # Open a file dialog to select the export location
                    # filename = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
                    # filename = str(log_inference) + 'test.pdf'
                    # LOGGER.info(str(log_inference))
                    # Generate the PDF report

                    # current_date = datetime.now()
                    # formatted_date = current_date.strftime('%Y-%m-%d')

                    file_path = str(log_inference) + '/' + formatted_date + '_log.TXT'
                    
                    content = ''
                    with open(file_path, 'r') as z:
                        content = z.readlines()
                        content = content[no_line_log]

                    LOGGER.info(content)

                    generate_report(content, Path(os.getcwd() + '/img_inference'))

                    raise StopIteration
                
                if datetime.now(tz=tzInfo) > lastDate:
                    lastDate = datetime.now(tz=tzInfo) + timedelta(seconds=timer, minutes=0, hours=0)
                    try:
                        if count_now != 0:
                            save_log(str(countUnripe) + ";" + str(countRipe)+ ";" + str(countOverripe) + ";" + str(countEmptybunch) + ";" + str(countAbnormal) + ";" + str(countKastrasi)  + ";"   + str(countLongStalk) +  ";"+ str(datetime.now(tz=tzInfo).strftime("%Y-%m-%d %H:%M:%S")), str(Path(log_dir)))
                            count_now = 0
                            countUnripe = 0
                            countRipe = 0
                            countOverripe = 0
                            countEmptybunch = 0
                            countAbnormal = 0
                            countLongStalk = 0
                            countKastrasi = 0
                            LOGGER.error("Data sudah disimpan")
                    except:
                        LOGGER.error("internet e mati bos")

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

            date = "/img_inference/" + str(datetime.now(tz=tzInfo).strftime("%Y-%m-%d"))
            
            # if time.time() > last_mins:
            #     if badCount > 5 and not bad_ffb:
            #         save_img_inference_sampling(im0, str(Path(os.getcwd())),'/img_inference/_worst_')
            #         bad_ffb = True
            #     elif goodCount > 3 and not good_ffb:
            #         save_img_inference_sampling(im0, str(Path(os.getcwd())), '/img_inference/_best_')
            #         good_ffb = True
            #     if good_ffb and bad_ffb:
            #         last_mins = time.time() + (mins * 60)
            #         good_ffb = False
            #         bad_ffb = False

            # if time.time() > last_mins:
            #     if badCount > 5 and not bad_ffb:
            #         save_img_inference_sampling(im0, str(Path(os.getcwd())),'/img_inference/_worst_')
            #         bad_ffb = True
            #     elif goodCount > 3 and not good_ffb:
            #         save_img_inference_sampling(im0, str(Path(os.getcwd())), '/img_inference/_best_')
            #         good_ffb = True
            #     if good_ffb and bad_ffb:
            #         last_mins = time.time() + (mins * 60)
            #         good_ffb = False
            #         bad_ffb = False
                
            

            # LOGGER.info(date + '_worst_')
                # if worstCount > 3 and not worst_ffb:
                #     save_img_inference_sampling(im0, str(Path(os.getcwd())), date + '_worst_')
                #     worst_ffb = True
                # elif bestCount > 3 and not best_ffb:
                #     save_img_inference_sampling(im0, str(Path(os.getcwd())),  date + '_best_')
                #     best_ffb = True
                # if best_ffb and worst_ffb:
                #     last_mins = time.time() + (mins * 60)
                #     best_ffb = False
                #     worst_ffb = False

            # print("badCount:" +str(badCount) + " | bad status=" + str(bad_ffb))
            # print("goodCount:" + str(goodCount) + " | good status=" + str(good_ffb))


    # Print results
    log_path = str(Path(save_dir))
    if os.path.isdir(log_path) != True:
       os.file(log_path) 
    f = open(log_path + '/log_hasil.csv' , "a")
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)
    model_list = str(yolo_model_str).split('/')
    #f.write('\n' + str(model_list[-2]))
    count_real = ''
    acc = 0.0
    f.write('\n' + str(model_list[-2]) + "," + str(imgsz) + "," + str(confidence) + "," + str(roi) + "," + str(iou) + "," + str(unripe) + "," + str(ripe) + str(overripe) + "," + str(empty_bunch)+ "," + str(abnormal) 
    + "," +  str(acc) + "," + (f'%.1f,%.1f,%.1f,%.1f' % t))

    f.close()
    if save_txt or save_vid:
        print('Results saved to %s' % save_path)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)
    

def save_img(img, save_dir, name):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgP = Image.frombytes("RGB", (int(img.shape[1]), int(img.shape[0])), img)
    myHeight, myWidth = imgP.size
    imgP = imgP.resize((myHeight, myWidth))
    imgP.save(save_dir +name+".JPG", optimize=True, quality=25)

def save_img_inference_sampling(img, save_dir, name):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgP = Image.frombytes("RGB", (int(img.shape[1]), int(img.shape[0])), img)
    myHeight, myWidth = imgP.size
    imgP = imgP.resize((myHeight, myWidth))
    imgP.save(save_dir +name+".JPG", optimize=True, quality=25)

def count_obj(box ,w , h, id, cls, folder_inference):
    global count, data, count_now, unripe, ripe, overripe, empty_bunch, abnormal, kastrasi, long_stalk, countUnripe, countRipe, countOverripe, countEmptybunch, countAbnormal, countLongStalk, countKastrasi, prctgUR, prctgRP, prctgOV, prctgEB, prctgAB, prctgKS, prctgLS
    center_coordinates = (int(box[0] + (box[2]-box[0])/2), int (box[1] + (box[3]-box[1])/2))
    wideArea = (box[2] - box[0] ) * (box[3] - box[1])
    if int (box[1] + (box[3]-box[1])/2) < (int(h*(1.0-roi))) and id not in data and id not in id_ffb:
        id_ffb.append(id)
        time_ffb.append(time.time())
    if int (box[1] + (box[3]-box[1])/2) > (int(h*(1.0-roi))):
        if id not in data:
            for x in id_ffb:
                if int(id) == int(x):
                    ffb_index = id_ffb.index(x)
                    id_ffb.remove(id)
                    time_ffb.pop(ffb_index)
                    count_now += 1

                    if cls != 5:
                        count += 1
                 
                    match cls:
                        case 0:
                            unripe += 1
                            countUnripe +=1
                        case 1:
                            ripe += 1
                            countRipe +=1
                        case 2:
                            overripe += 1
                            countOverripe +=1
                        case 3:
                            empty_bunch += 1
                            countEmptybunch +=1
                        case 4:
                            abnormal += 1
                            countAbnormal +=1
                        case 5:
                            long_stalk += 1
                            countLongStalk +=1
                        case 6:
                            kastrasi += 1
                            countKastrasi +=1
                            # save_kastrasi(str(kastrasi) + " " + str(f'{wideArea:,}'), folder_kastrasi)

                    # save_inference_data(str(unripe) + ";" + str(ripe)+ ";" + str(overripe) + ";" + str(empty_bunch) + ";" + str(abnormal) + ";" + str(kastrasi) + ";" + str(dateStart) + ";", str(folder_inference))
                    try:
                        prctgUR = round((unripe/ count) * 100,2)
                        prctgRP = round((ripe/ count) * 100,2)
                        prctgOV = round((overripe/ count) * 100,2)
                        prctgEB = round((empty_bunch/ count) * 100,2)
                        prctgAB = round((abnormal/ count) * 100,2)
                        prctgKS = round((kastrasi/ count) * 100,2)
                        prctgLS = round((long_stalk/ count) * 100,2)
                    except:
                        LOGGER.info('Jumlah count = 0')
                  

                    data.append(id)
                    for z in time_ffb:
                        if time.time() > float(z) + float(6):
                            ffb_index = time_ffb.index(z)
                            id_ffb.pop(ffb_index)
                            time_ffb.remove(z)
                    break
                
        # print(str(id_ffb))
        # print(str(time_ffb))

            
# async def post_count():
#     global count_now
#     async with aiohttp.ClientSession() as session:
#         params = {'count': str(count_now), 'timestamp': datetime.now(tz=tzInfo).strftime("%Y-%m-%d %X")}
#         async with session.post(url,data=params) as resp:
#             count_now = 0
#             response = await resp.read()
#             LOGGER.info('Response status ' + str(response))

def save_inference_data(header, path):
    global path_no_plat, no_line_log, path_plat
    header_str = str(header) 
    # LOGGER.info('sudah disini gan')
    # file_path = path + '/log_inference.TXT'
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # if not os.path.exists(file_path):
    #     f = open(file_path, "a")
    #     f.write("")
    #     f.close()
    # LOGGER.info('62')

    path_inference_truk = path +'/' +formatted_date + '_log.TXT'
    
    

    with open(path_inference_truk, 'r') as x:
        content = x.readlines()
        
        count = 0
        # LOGGER.info('63')
        for line in content:
            if 'Sedang Berjalan' in line:
                # LOGGER.info('65')
                targetLine = content[count]
                wr = open(path_inference_truk, "w")
                content[count] = targetLine.replace(";Sedang Berjalan","")
                wr.writelines(content)
                wr.close()

                strData = content[count].split(';')
                no_plat = strData[1]

                
                path_no_plat = path + '/' + no_plat + '.TXT'
                LOGGER.info(path_no_plat)
                path_plat = strData[1]
                no_line_log = count
               
                # # LOGGER.info(header_str)
                # # time.sleep(5) 
                # with open(file_path, 'r') as y:
                #     inferenceData = y.readlines()
                #     # LOGGER.info(count)
                #     # LOGGER.info(inferenceData)
                #     targetLine2 = inferenceData[count]

                #     wr2 = open(file_path, "w")
                #     inferenceData[count] =  header_str
                #     wr2.close()

            count += 1          

    if not os.path.exists(path):
                    os.makedirs(path)
    if not os.path.exists(path_no_plat):
            # LOGGER.info('66')
            f = open(path_no_plat, "a")
            f.write("")
            f.close()
    with open(path_no_plat, 'r') as z:
        content = z.readlines()
        # LOGGER.info('67')
        wr = open(path_no_plat, "w")
        try:
            # LOGGER.info('68')
            if len(content[0].strip()) == 0 | content[0] in ['\n', '\r\n']:
                wr.write(header_str)
                # LOGGER.info('69')
            else:
                wr.write(header_str)
        except:
            # LOGGER.info('70')
            wr.write(header_str)
        wr.close()

def delete_inference_file(path_plat):
    # LOGGER.info('60')
    os.remove(path_plat)
    
def append_inference_data(header, path, no_line):
    header_str = str(header)
    # LOGGER.info(header_str)
    
    file_path = path + '/' + formatted_date + '_log.TXT'
    file_path_inference = path + '/log_inference.TXT'
    
    with open(file_path, 'r') as z:
        content = z.readlines()

        content[no_line] = content[no_line].rstrip("\n") + header_str + '\n'

        with open(file_path,"w") as file:
            file.writelines(content)
      

        wr = open(file_path_inference, "a")
        try:
            if len(content[0].strip()) == 0 | content[0] in ['\n', '\r\n']:
                wr.write(content[no_line])
            else:
                wr.write(content[no_line])
        except:
            wr.write(content[no_line])
        wr.close()
       
        # wr = open(file_path, "a")
        # try:
        #     if len(content[0].strip()) == 0 | content[0] in ['\n', '\r\n']:
        #         wr.write(header_str)
        #     else:
        #         wr.write(header_str)
        # except:
        #     wr.write(header_str)
         
       
        # content[no_line] = content[no_line] +
        # wr.write(content) 
        # wr.close()

def save_log(header, path):
    header_str = str(header)  + '\n'
    file_path = path + '/log.TXT'
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(file_path):
        f = open(file_path, "a")
        f.write("")
        f.close()
    with open(file_path, 'r') as z:
        content = z.readlines()
        wr = open(file_path, "a")
        try:
            if len(content[0].strip()) == 0 | content[0] in ['\n', '\r\n']:
                wr.write(header_str)
            else:
                wr.write(header_str)
        except:
            wr.write(header_str)
        wr.close()

def save_kastrasi(header, path):
    header_str = str(header)  + '\n'
    file_path = path + '/log.TXT'
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(file_path):
        f = open(file_path, "a")
        f.write("")
        f.close()
    with open(file_path, 'r') as z:
        content = z.readlines()
        wr = open(file_path, "a")
        try:
            if len(content[0].strip()) == 0 | content[0] in ['\n', '\r\n']:
                wr.write(header_str)
            else:
                wr.write(header_str)
        except:
            wr.write(header_str)
        wr.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf_thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--roi', type=float, default=0.3, help='line height')
    opt = parser.parse_args()
    roi = opt.roi
    confidence = opt.conf_thres
    yolo_model_str = opt.yolo_model
    iou = opt.iou_thres
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    
    #os.system('python send_query.py')

    with torch.no_grad():
        detect(opt)
