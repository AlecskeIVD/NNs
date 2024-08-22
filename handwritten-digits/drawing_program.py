import pygame as pg
import numpy as np

pg.init()
WIDTH, HEIGHT = 1440, 770
WINDOW = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("Drawing digits")
TOP_LEFT_X, TOP_LEFT_Y = 125, 50
FPS = 120
RESOLUTION = 5
SQUARE_SIZE = 24 // RESOLUTION


def generate_matrix2(drawing):
    output = np.zeros((28, 28), np.int16)
    factor = 255 / (RESOLUTION**2)
    for i in range(28):
        for j in range(28):
            temp = 0
            for a in range(RESOLUTION):
                for b in range(RESOLUTION):
                    temp += drawing[i*RESOLUTION+a][j*RESOLUTION+b]
            output[i, j] = round(temp * factor)
    return output


def generate_matrix(drawing):
    drawing_np = np.array(drawing, dtype=np.int16)
    factor = 255 / (RESOLUTION ** 2)
    output = np.zeros((28, 28), np.int16)
    for i in range(28):
        for j in range(28):
            block = drawing_np[i * RESOLUTION:(i + 1) * RESOLUTION, j * RESOLUTION:(j + 1) * RESOLUTION]
            output[i, j] = round(block.sum() * factor)
    return output


def main():
    draw_mode = False
    eraser_mode = False
    passive_mode = True
    run = True
    clock = pg.time.Clock()
    drawing = [[(i + j) % 2 for i in range(28*RESOLUTION)] for j in range(28*RESOLUTION)]
    input_NN = generate_matrix(drawing)
    print(input_NN)
    BRUSH_SIZE = RESOLUTION // 2
    while run:
        clock.tick(FPS)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False

            if event.type == pg.KEYDOWN:
                if event.key == pg.K_p:
                    # Enter passive mode
                    draw_mode = False
                    eraser_mode = False
                    passive_mode = True
                if event.key == pg.K_d:
                    # Enter draw mode
                    draw_mode = True
                    eraser_mode = False
                    passive_mode = False
                if event.key == pg.K_e:
                    # Enter eraser mode
                    draw_mode = False
                    eraser_mode = True
                    passive_mode = False
                if event.key == pg.K_c:
                    # clear board + enter passive mode
                    draw_mode = False
                    eraser_mode = False
                    passive_mode = True
                    drawing = [[0 for i in range(28*RESOLUTION)] for j in range(28*RESOLUTION)]
                elif event.unicode.isdigit():  # Only allow numeric input
                    BRUSH_SIZE = int(event.unicode)
        draw(WINDOW, drawing)
        pg.display.update()
        mouse_pos = pg.mouse.get_pos()

        if not passive_mode and (TOP_LEFT_X < mouse_pos[0] < TOP_LEFT_X+len(drawing[0])*SQUARE_SIZE) and (TOP_LEFT_Y < mouse_pos[1] < TOP_LEFT_Y+len(drawing)*SQUARE_SIZE):
            translated_mouse_x, translated_mouse_y = mouse_pos[0]-TOP_LEFT_X, mouse_pos[1]-TOP_LEFT_Y
            j, i = translated_mouse_x // SQUARE_SIZE, translated_mouse_y // SQUARE_SIZE
            if i < len(drawing) and j < len(drawing[0]):
                if eraser_mode and drawing[i][j] == 1:
                    drawing[i][j] = 0
                if draw_mode and drawing[i][j] == 0:
                    drawing[i][j] = 1
            for n in range(max(i-BRUSH_SIZE, 0), min(i+BRUSH_SIZE+1, len(drawing))):
                for m in range(max(j-BRUSH_SIZE, 0), min(j+BRUSH_SIZE+1, len(drawing[0]))):
                    if n != i or m != j:
                        if eraser_mode and drawing[n][m] == 1:
                            drawing[n][m] = 0
                        if draw_mode and drawing[n][m] == 0:
                            drawing[n][m] = 1
            input_NN = generate_matrix(drawing)
            assert np.array_equal(input_NN, generate_matrix2(drawing))
            print("array: ", input_NN)

    pg.quit()


def draw(window: pg.surface, drawing):
    window.fill((0, 0, 55))
    for row in range(len(drawing)):
        for column in range(len(drawing[row])):
            if drawing[row][column] == 1:
                pg.draw.rect(window, (255, 255, 255), (TOP_LEFT_X+column*SQUARE_SIZE, TOP_LEFT_Y + row*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
    input_NN = generate_matrix(drawing)

    # We draw image 112x112
    size = int(112/28)
    for i in range(28):
        for j in range(28):
            val = input_NN[i, j]
            pg.draw.rect(window, (val, val, val),
                         (5+j * size,5+ i * size, size, size))


    # DRAW LINES TO IMPROVE CLARITY BOARD
    for row in range(len(drawing)+1):
        pg.draw.line(window, (0, 0, 0), (TOP_LEFT_X+row*SQUARE_SIZE, TOP_LEFT_Y), (TOP_LEFT_X+row*SQUARE_SIZE, TOP_LEFT_Y+len(drawing)*SQUARE_SIZE))

    for column in range(len(drawing[0])+1):
        pg.draw.line(window, (0, 0, 0), (TOP_LEFT_X, TOP_LEFT_Y+column*SQUARE_SIZE), (TOP_LEFT_X+len(drawing[0])*SQUARE_SIZE, TOP_LEFT_Y+column*SQUARE_SIZE))

    font = pg.font.Font(None, 18)
    text_surface = font.render(" Choose between (d)raw mode, (e)raser mode, (p)assive mode or (c)learing the board", True, (255, 255, 255))  # White text
    text_rect = text_surface.get_rect()

    # Position the text under the board
    text_rect.center = (TOP_LEFT_X + len(drawing[0]) * SQUARE_SIZE // 2,
                        TOP_LEFT_Y + len(drawing) * SQUARE_SIZE + 20)

    # Draw the text on the window
    window.blit(text_surface, text_rect)


if __name__ == "__main__":
    main()
