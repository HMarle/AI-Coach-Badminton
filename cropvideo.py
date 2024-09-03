import os
from moviepy.editor import VideoFileClip

def resize_and_crop_videos(input_folder, output_folder, crop_top, crop_sides):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            with VideoFileClip(input_path) as video:
                width, height = video.size
                new_width = width - 2 * crop_sides
                new_height = height - crop_top

                cropped_video = video.crop(x1=crop_sides, y1=crop_top, x2=width-crop_sides, y2=height)
                cropped_video.write_videofile(output_path, codec='libx264')

if __name__ == "__main__":
    input_folder = ''  # Remplacer par le chemin du dossier contenant les vidéos
    output_folder = ''  # Remplacer par le chemin du dossier de sortie
    crop_top = 180  # Nombre de pixels à couper en haut
    crop_sides = 200  # Nombre de pixels à couper de chaque côté

    resize_and_crop_videos(input_folder, output_folder, crop_top, crop_sides)
