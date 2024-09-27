#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--mirror', action='store_true')

    parser.add_argument(
        "--model",
        type=str,
        default=
        '02.model/DeepLabV3Plus(timm-mobilenetv3_small_100)_1366_2.16M_0.8297/best_model_simplifier.onnx'
    )
    parser.add_argument("--score", type=float, default=0.5)

    args = parser.parse_args()

    return args


def run_inference(onnx_session, input_size, image):
    # 前処理
    input_image = cv.resize(image, dsize=(input_size, input_size))  # リサイズ
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)  # BGR→RGB変換

    # 標準化
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (input_image / 255 - mean) / std

    # HWC → CHW
    x = x.transpose(2, 0, 1).astype('float32')

    # (1, 3, Height, Width)形式へリシェイプ
    x = x.reshape(-1, 3, 512, 512)

    # 推論
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})
    onnx_result = np.array(onnx_result).squeeze()

    return onnx_result


def main():
    # 引数解析 #################################################################
    args = get_args()
    cap_device = args.device
    image_path = args.image
    cap_width = args.width
    cap_height = args.height

    if args.video is not None:
        cap_device = args.video

    mirror = args.mirror

    model = args.model
    score_th = args.score

    # カメラ準備 ###############################################################
    if image_path is None:
        cap = cv.VideoCapture(cap_device)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

        _, frame = cap.read()
    else:
        frame = cv.imread(image_path)

    # モデルロード #############################################################
    input_size = 512
    onnx_session = onnxruntime.InferenceSession(
        model,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    )

    # 背景画像リスト ###########################################################
    bg_image_list = []

    # クラスID：0
    bg_image = cv.imread('XX.image/01.png')
    bg_image_list.append(bg_image)

    # クラスID：1
    bg_image = cv.imread('XX.image/02.png')
    bg_image_list.append(bg_image)

    # クラスID：2
    bg_image = cv.imread('XX.image/03.png')
    bg_image_list.append(bg_image)

    if image_path is None:
        while True:
            start_time = time.time()

            # カメラキャプチャ #####################################################
            ret, frame = cap.read()
            if not ret:
                break
            if mirror:
                frame = cv.flip(frame, 1)  # ミラー表示
            debug_image = copy.deepcopy(frame)

            # 検出実施 ##############################################################
            masks = run_inference(
                onnx_session,
                input_size,
                frame,
            )

            # 閾値判定
            masks = np.where(masks > score_th, 0, 1)

            elapsed_time = time.time() - start_time

            # デバッグ描画
            debug_image = draw_debug(
                debug_image,
                elapsed_time,
                masks,
                bg_image_list,
            )

            # キー処理(ESC：終了) ##################################################
            key = cv.waitKey(1)
            if key == 27:  # ESC
                break

            # 画面反映 #############################################################
            cv.imshow('Demo', debug_image)

        cap.release()
    else:
        debug_image = copy.deepcopy(frame)

        # ウォームアップ
        _ = run_inference(
            onnx_session,
            input_size,
            frame,
        )

        start_time = time.time()

        # 検出実施 ##############################################################
        masks = run_inference(
            onnx_session,
            input_size,
            frame,
        )

        # 閾値判定
        masks = np.where(masks > score_th, 0, 1)

        elapsed_time = time.time() - start_time

        # デバッグ描画
        debug_image = draw_debug(
            debug_image,
            elapsed_time,
            masks,
            bg_image_list,
        )

        # 画面反映 #############################################################
        cv.imshow('Demo', debug_image)
        key = cv.waitKey(-1)

    cv.destroyAllWindows()


def draw_debug(
        image,
        elapsed_time,
        masks,
        bg_image_list,
        border_color=(0, 0, 0),
):
    image_width, image_height = image.shape[1], image.shape[0]

    debug_image = copy.deepcopy(image)

    if len(masks) > 1:
        for index, mask in enumerate(masks):
            if bg_image_list[index] is None:
                continue

            # 背景画像
            bg_image = cv.resize(bg_image_list[index],
                                 (image_width, image_height))

            # マスク画像
            mask = np.stack((mask, ) * 3, axis=-1).astype('uint8')
            resize_mask = cv.resize(mask, (image_width, image_height))

            # 枠線描画
            if border_color is not None:
                frame_border_image = np.zeros(image.shape, dtype=np.uint8)
                frame_border_image[:] = border_color

                kernel = np.ones((5, 5), np.uint8)
                resize_mask2 = cv.erode(resize_mask, kernel, iterations=1)

                debug_image = np.where(resize_mask2, debug_image,
                                       frame_border_image)

            # 各クラスマスク描画
            debug_image = np.where(resize_mask, debug_image, bg_image)
    else:
        if bg_image_list[0] is not None:
            # 背景画像
            bg_image = cv.resize(bg_image_list[0], (image_width, image_height))

            # マスク画像
            mask = np.stack((masks, ) * 3, axis=-1).astype('uint8')
            resize_mask = cv.resize(mask, (image_width, image_height))

            # 枠線描画
            if border_color is not None:
                frame_border_image = np.zeros(image.shape, dtype=np.uint8)
                frame_border_image[:] = border_color

                kernel = np.ones((5, 5), np.uint8)
                resize_mask2 = cv.erode(resize_mask, kernel, iterations=1)

            debug_image = np.where(resize_mask2, debug_image,
                                   frame_border_image)

            # 各クラスマスク描画
            debug_image = np.where(resize_mask, debug_image, bg_image)

    # 処理時間
    cv.putText(debug_image,
               "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4,
               cv.LINE_AA)
    cv.putText(debug_image,
               "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,
               cv.LINE_AA)

    return debug_image


if __name__ == '__main__':
    main()
