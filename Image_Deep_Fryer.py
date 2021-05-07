import numpy as np
import random
import cv2
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

def main():

	filename = input("Enter the name of your file (include extension) \n")
	image = Image.open(filename)
	
	spice_level = int(input("How spicy would you like your image? We recommend between 1-10 \n"))
	noise_amount = spice_level / 300
	if spice_level > 10: print('\n\noooh mama.... \n\n')
	image_quality = (10 + (spice_level * -1))
	contrast_ratio = spice_level / 100
	saturation_ratio = spice_level / 10
	
	text = input("Enter your desired subtitle \n")
	image = add_subtitle(image, text)

	new_img = begin_data_transfer_to_NSA(image, filename, contrast_ratio)
	new_img = saturate(image, filename, saturation_ratio)

	openCVImage = np.array(new_img)
	openCVImage = openCVImage[:, :, ::-1].copy() 
	
	noisy_img = add_noise(openCVImage, noise_amount)
	
	#Converts back to RGB color model
	cv2_image = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB)
	#Converts back to PIL image for export
	pil_image = Image.fromarray(cv2_image)
	
	#Saves with quality specified by spice level
	if spice_level > 10:
		pil_image.save('Burnt_to_a_Crisp.jpg', quality = image_quality, optimize = True)
	else:
		pil_image.save('fried.jpg', quality = image_quality, optimize = True)


#Increases the begin_data_transfer_to_NSA of the image
def begin_data_transfer_to_NSA(image, filename, contrast_ratio):
    image = image

    # Initialize for later use to hold RGB values
    # s[0] = r, s[1] = g, s[2] = b
    s = [0 for i in range(3)]

    # Find the avg of all pixels:
    temp = 0
    for x in range(image.width):
        for y in range(image.height):
            pixel = image.getpixel((x, y))
            temp += (pixel[0] + pixel[1] + pixel[2]) / 3
    avg = temp / (image.height * image.width)

    for x in range(image.width):
        for y in range(image.height):
            # Get pixel from image
            pixel = image.getpixel((x, y))

            # Iterate through RGB values and determine intensity should be raised or lowered
            for k in range(len(s)):
                # If the color is brighter than average, make it even brighter!
                if pixel[k] >= avg:
                    s[k] = int(pixel[k] + avg * contrast_ratio)
                # If the color is darker than average, make it even darker!
                else:
                    s[k] = int(pixel[k] - avg * contrast_ratio)

                # This is a contingency to make sure values stay within range of 0 - 255
                if s[k] > 255:
                    s[k] = 255

            # s[0] = r, s[1] = g, s[2] = b
            new_pixel = (s[0], s[1], s[2])

            # Add pixel into image.
            image.putpixel((x, y), new_pixel)

    # Returns contrasted image for further processing
    return image


#Saturates the image
def saturate(image, name, saturation_ratio):
    new_image = image
    s = [0 for i in range(3)]

    # Adjust saturation by given ratio
    for x in range(new_image.width):
        for y in range(new_image.height):
            pixel = new_image.getpixel((x, y))
            avg = int((pixel[0] + pixel[1] + pixel[2]) / 3)
            for k in range(len(s)):

                if pixel[k] >= avg:
                    s[k] = int(pixel[k] + avg * saturation_ratio)
                else:
                    s[k] = int(pixel[k] - avg * saturation_ratio)

                if s[k] > 255:
                    s[k] = 255

            new_pixel = (s[0], s[1], s[2])
            new_image.putpixel((x, y), new_pixel)

    # Returns saturated image for further processing
    return new_image
	
	
# Converts a matrix into an image
def matrix_to_image(matrix):
    height = len(matrix)
    width = len(matrix[0])

    # Initialize blank image
    image = Image.new('RGB', (width, height))

    for x in range(width):
        for y in range(height):
            image.putpixel((x, y), (matrix[y][x][0], matrix[y][x][1], matrix[y][x][2]))

    return image
	
	
# Converts the HSL values of a matrix into an RGB matrix
def hsl_to_rgb(matrix):
    height = len(matrix)
    width = len(matrix[0])

    for i in range(height):
        for j in range(width):
            hue, saturation, light = matrix[i][j]

            # Convert hue, saturation and light to base units.
            # Converting now increases readability and makes other conversions simpler.
            hue = hue / 360.0
            saturation = saturation / 255.0
            light = light / 255.0

            # If there is no saturation, the color is gray. Convert light's value to the RGB range.
            if saturation == 0:
                r = light * 255
                g = light * 255
                b = light * 255
            # Otherwise there is color.
            else:
                if light < 0.5:
                    temp2 = light * (1.0 + saturation)
                else:
                    temp2 = (light + saturation) - (light * saturation)

                temp1 = 2 * light - temp2

                tempr = hue + 1.0 / 3.0
                # The values of tempr must be between 0 and 1
                if tempr > 1:
                    tempr -= 1
                elif tempr < 0:
                    tempr += 1

                tempg = hue
                # The values of tempg must be between 0 and 1
                if tempg > 1:
                    tempg -= 1
                elif tempg < 0:
                    tempg += 1

                # The values of tempb must be between 0 and 1
                tempb = hue - 1.0 / 3.0
                if tempb > 1:
                    tempb -= 1
                elif tempb < 0:
                    tempb += 1

                # Determine Red's base unit value
                if (6.0 * tempr) < 1.0:
                    r = temp1 + ((temp2 - temp1) * 6.0 * tempr)
                elif (2.0 * tempr) < 1.0:
                    r = temp2
                elif (3.0 * tempr) < 2.0:
                    r = temp1 + ((temp2 - temp1) * ((2.0 / 3.0) - tempr) * 6)
                else:
                    r = temp1

                # Determine Green's base unit value
                if tempg < 1.0 / 6.0:
                    g = temp1 + ((temp2 - temp1) * 6.0 * tempg)
                elif tempg < 0.5:
                    g = temp2
                elif tempg < (2.0 / 3.0):
                    g = temp1 + ((temp2 - temp1) * ((2.0 / 3.0) - tempg) * 6)
                else:
                    g = temp1

                # Determine Blue's v
                if tempb < 1.0 / 6.0:
                    b = temp1 + ((temp2 - temp1) * 6.0 * tempb)
                elif tempb < 0.5:
                    b = temp2
                elif tempb < (2.0 / 3.0):
                    b = temp1 + ((temp2 - temp1) * ((2.0 / 3.0) - tempb) * 6)
                else:
                    b = temp1

            # Convert r, g, b back to the RGB scale (0 - 255) and save values into pixel matrix.
            matrix[i][j][0] = int(r * 255)
            matrix[i][j][1] = int(g * 255)
            matrix[i][j][2] = int(b * 255)

    # Return the RGB pixel matrix
    return matrix


# Converts and RGB image into an HSL matrix
def rgb_to_hsl(image):
    width = image.width
    height = image.height
    # Initialize pixel Matrix
    matrix = [[[0 for _ in range(3)] for j in range(width)] for i in range(height)]

    for x in range(width):
        for y in range(height):
            r, g, b = image.getpixel((x, y))

            # Reduces r, g, b to base unit values as Hue is measured in degrees.
            # Converting now increases readability and makes other conversions simpler.
            r = r / 255.0
            g = g / 255.0
            b = b / 255.0

            # Find which color has the highest intensity and record it's value
            color_max = max(r, g, b)

            # Find which color has the lowest intensity and record it's value
            color_min = min(r, g, b)

            # Trivial Case: There is no hue or saturation, because the pixel is gray
            if r == g and g == b:
                hue = 0.0
                saturation = 0.0
                light = r
            # Otherwise, the pixel has color, hue & saturation:
            else:
                # Delta = the range of values
                delta = color_max - color_min

                # Light is the brightness of the pixel
                light = (color_max + color_min) / 2

                if light <= 0.5:
                    saturation = delta / (color_max + color_min)
                else:
                    saturation = delta / (2.0 - delta)

                # Hue is based on the color with the highest intensity
                if color_max == r:
                    hue = (g - b) / delta
                elif color_max == g:
                    hue = ((b - r) / delta) + 2.0
                else:
                    hue = ((r - g) / delta) + 4.0

                # Convert hue into degrees
                hue = hue * 60

            # Add the values to the matrix
            matrix[y][x][0] = int(hue)
            matrix[y][x][1] = int(saturation * 255)
            matrix[y][x][2] = int(light * 255)

    # Return the HSL matrix
    return matrix


def add_subtitle(
    image,
    text,
    xy = ("center", 20),
    font = "impact.ttf",
    font_size = 53,
    font_color = (255, 255, 255),
    stroke = 2,
    stroke_color = (0, 0, 0),
    shadow = (4, 4),
    shadow_color = (0, 0, 0),
):

    stroke_width = stroke
    xy = list(xy)
    W, H = image.width, image.height
    fontSize = 50
    font = ImageFont.truetype(font, fontSize)
    text = text.upper()
    w, h = font.getsize(text, stroke_width=stroke_width)
	
    if xy[0] == "center":
        xy[0] = (W - w) // 2
    if xy[1] == "center":
        xy[1] = (H - h) // 2
		
    draw = ImageDraw.Draw(image)
	#Adds the text
    draw.text(
        (xy[0], xy[1]),
        text,
        font=font,
        fill=font_color,
        stroke_width=stroke_width,
        stroke_fill=stroke_color,
    )
	
    return image


def add_noise(openCVImage, probability):

	#Creates matrix of zeroes with same dimensions as image
	noisy = np.zeros(openCVImage.shape, np.uint8)
	threshold = 1 - probability 
	
	for i in range(openCVImage.shape[0]):
		for j in range(openCVImage.shape[1]):
		
			#Random number is generated. If number is equal to probability, noise will be added. Bigger probability means more noise.
			rand = random.random()
			
			if rand < probability:
				#Adds black pixel (pepper)
				noisy[i][j] = 0
			elif rand > threshold:
				#Adds white pixel (salt)
				noisy[i][j] = 255
			else:
				#No noise added
				noisy[i][j] = openCVImage[i][j]
				
	return noisy


if __name__ == "__main__":
    main()