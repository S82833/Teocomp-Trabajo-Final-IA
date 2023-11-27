import pygame, sys
import random
import tensorflow as tf
import numpy as np

# Inicializar pygame
pygame.init()

# Constantes
WIDTH = 800
HEIGHT = 800
PLAYER_SIZE = 50
PLAYER_SPEED = 5
PLAYER_JUMP = 100
JUMP_STRENGTH = -15
GRAVITY = 1
ENEMY_SIZE = 40
ENEMY_SPEED = 2
PLATFORM_HEIGHT = 20
PLATFORM_WIDTH = 150
PLAYER_COLOR = (0, 255, 0)
ENEMY_COLOR = (255, 0, 0)
ENEMY_VISION_RANGE = 100
ENEMY_INCREMENT_SPEED = 0.1
SAFE_ZONE = 150
VISION_RANGE = 150
POWERUP_SIZE = 30
POWERUP_COLOR = (255, 255, 0)
POWERUP_DURATION = 2000
CURRENT_LEVEL = 1

# Configurar la ventana
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Automaton Runner')

background_image = pygame.image.load("square/fondo_b.png")

# Clase para el jugador
class Player:
    def __init__(self):
        self.x = WIDTH // 2
        self.y = HEIGHT - PLAYER_SIZE - 10
        self.dx = 0
        self.dy = 0
        self.jumping = False
        self.on_ground = True
        self.times_hit = 0
        self.powered_up = False
        self.powerup_start_time = 0
        self.collected_powerup = False

    def activate_powerup(self):
        """Activa el power-up."""
        self.powered_up = True
        self.powerup_start_time = pygame.time.get_ticks()
        self.collected_powerup = True

    def deactivate_powerup(self):
        """Desactiva el power-up."""
        self.powered_up = False

    def check_powerup_duration(self):
        """Verifica la duración del power-up y lo desactiva si ha expirado."""
        if self.powered_up and pygame.time.get_ticks() - self.powerup_start_time > POWERUP_DURATION:
            self.deactivate_powerup()

    def move(self, platforms):
        """Mueve al jugador y maneja las colisiones con las plataformas."""
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.dx = -PLAYER_SPEED
        elif keys[pygame.K_RIGHT]:
            self.dx = PLAYER_SPEED
        else:
            self.dx = 0

        if keys[pygame.K_SPACE] and self.on_ground:
            self.dy = JUMP_STRENGTH
            self.jumping = True
            self.on_ground = False

        # Aplicar gravedad
        self.dy += GRAVITY

        # Colisiones con plataformas
        for platform in platforms:
            platform_rect = pygame.Rect(platform[0], platform[1], PLATFORM_WIDTH, PLATFORM_HEIGHT)
            if platform_rect.colliderect(self.x, self.y + self.dy, PLAYER_SIZE, PLAYER_SIZE):
                if self.dy > 0:
                    self.dy = 0
                    self.y = platform[1] - PLAYER_SIZE
                    self.on_ground = True
                    self.jumping = False

        self.x += self.dx
        self.y += self.dy

        # Mantener al jugador dentro de la ventana y en el suelo
        self.x = max(0, min(WIDTH - PLAYER_SIZE, self.x))
        if self.y > HEIGHT - PLAYER_SIZE:
            self.y = HEIGHT - PLAYER_SIZE
            self.dy = 0
            self.on_ground = True
            self.jumping = False

    def draw(self, screen):
        """Dibuja al jugador en la pantalla."""
        if self.powered_up:
            color = (0, 0, 255)
        else:
            color = PLAYER_COLOR
        pygame.draw.rect(screen, color, (self.x, self.y, PLAYER_SIZE, PLAYER_SIZE))

# Clase para el enemigo con aprendizaje
class LearningEnemy:
    def __init__(self):
        self.x = random.randint(0, WIDTH - ENEMY_SIZE)
        self.y = random.randint(0, HEIGHT - ENEMY_SIZE)
        self.dx = ENEMY_SPEED
        self.dy = ENEMY_SPEED
        self.state = "patrolling"
        self.model = self.build_model()


    def build_model(self):
        """Construye el modelo de la red neuronal para el enemigo."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(4,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(4, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model


    def initial_movement(self):
        """Movimiento inicial de patrullaje."""
        # El enemigo se mueve de lado a lado de manera predeterminada
        if self.state == "patrolling":
            # Al llegar a los límites, cambia la dirección
            if self.x <= 0 or self.x >= WIDTH - ENEMY_SIZE:
                self.dx = -self.dx

            # Mover solo si no se excede el ancho de la ventana
            if 0 <= self.x + self.dx <= WIDTH - ENEMY_SIZE:
                self.x += self.dx
                
    def move(self, player):
        self.initial_movement()
        
        """Mueve al enemigo y maneja su lógica de aprendizaje."""
        game_state = np.array([self.x, self.y, player.x, player.y])
        predicted_action = self.model.predict(np.expand_dims(game_state, axis=0))[0]
        self.dx = predicted_action[0] * ENEMY_SPEED
        self.dy = predicted_action[1] * ENEMY_SPEED

        # Calcular la distancia entre el enemigo y el jugador
        distance_x = player.x - self.x
        distance_y = player.y - self.y
        distance = np.linalg.norm([distance_x, distance_y])

        # Mantener una distancia normal del jugador
        normal_distance = 100  # Ajusta la distancia normal deseada

        if distance < normal_distance:
            # Calcular la dirección hacia el jugador
            direction_x = player.x - self.x
            direction_y = player.y - self.y
            distance_sum = abs(direction_x) + abs(direction_y)

            # Normalizar la dirección y ajustar el movimiento del enemigo
            if distance_sum != 0:
                direction_x /= distance_sum
                direction_y /= distance_sum

            # Calcular el nuevo movimiento sin salir de los límites
            new_x = self.x + direction_x * ENEMY_SPEED
            new_y = self.y + direction_y * ENEMY_SPEED

            # Verificar límites horizontales
            if 0 <= new_x <= WIDTH - ENEMY_SIZE:
                self.x = new_x
            else:
                self.dx = 0  # Detener el movimiento horizontal

            # Verificar límites verticales
            if 0 <= new_y <= HEIGHT - ENEMY_SIZE:
                self.y = new_y
            else:
                self.dy = 0  # Detener el movimiento vertical

    def draw(self, screen):
        """Dibuja al enemigo en la pantalla."""
        pygame.draw.rect(screen, ENEMY_COLOR, (self.x, self.y, ENEMY_SIZE, ENEMY_SIZE))

# Clase para el power-up
class PowerUp:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = POWERUP_SIZE

    def draw(self, screen):
        """Dibuja el power-up en la pantalla."""
        pygame.draw.circle(screen, POWERUP_COLOR, (self.x + self.size // 2, self.y + self.size // 2), self.size // 2)

class Button():
	def __init__(self, image, pos, text_input, font, base_color, hovering_color):
		self.image = image
		self.x_pos = pos[0]
		self.y_pos = pos[1]
		self.font = font
		self.base_color, self.hovering_color = base_color, hovering_color
		self.text_input = text_input
		self.text = self.font.render(self.text_input, True, self.base_color)
		if self.image is None:
			self.image = self.text
		self.rect = self.image.get_rect(center=(self.x_pos, self.y_pos))
		self.text_rect = self.text.get_rect(center=(self.x_pos, self.y_pos))

	def update(self, screen):
		if self.image is not None:
			screen.blit(self.image, self.rect)
		screen.blit(self.text, self.text_rect)

	def checkForInput(self, position):
		if position[0] in range(self.rect.left, self.rect.right) and position[1] in range(self.rect.top, self.rect.bottom):
			return True
		return False

	def changeColor(self, position):
		if position[0] in range(self.rect.left, self.rect.right) and position[1] in range(self.rect.top, self.rect.bottom):
			self.text = self.font.render(self.text_input, True, self.hovering_color)
		else:
			self.text = self.font.render(self.text_input, True, self.base_color)

SCREEN = pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("Menu")

BG = pygame.image.load("square/Background.png")

def get_font(size): # Returns Press-Start-2P in the desired size
    #Estilo de letra 
    return pygame.font.Font("square/font.ttf", size)

def play():
    while True:
        PLAY_BALL = Button(game_loop())
        PLAY_MOUSE_POS = pygame.mouse.get_pos()

        SCREEN.fill("black")

        #PLAY_TEXT = get_font(45).render("This is the PLAY screen.", True, "White")
        #PLAY_RECT = PLAY_BALL.get_rect(center=(200, 400))
        SCREEN.blit(PLAY_BALL)

        #Button BACK
        PLAY_BACK = Button(image=None, pos=(800, 800), 
                            text_input="BACK", font=get_font(20), base_color="White", hovering_color="Green")

        PLAY_BACK.changeColor(PLAY_MOUSE_POS)
        PLAY_BACK.update(SCREEN)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if PLAY_BACK.checkForInput(PLAY_MOUSE_POS):
                    main_menu()

        pygame.display.update()

def options():
    while True:
        OPTIONS_MOUSE_POS = pygame.mouse.get_pos()

        SCREEN.fill("black")

        OPTIONS_TEXT = get_font(30).render("*El juego consistira en la constante persecusion hacia el jugador", True, "white")
        OPTIONS_RECT = OPTIONS_TEXT.get_rect(center=(400, 250))
        OPTIONS_TEXT1 = get_font(30).render("*El jugador podra acabar con su enemigo consumiendo la pastilla amarilla", True, "white")
        OPTIONS_RECT1 = OPTIONS_TEXT1.get_rect(center=(400, 210))
        OPTIONS_TEXT2 = get_font(30).render("*Se podra pasar de nivel en la ultima plataforma", True, "white")
        OPTIONS_RECT2 = OPTIONS_TEXT2.get_rect(center=(400, 290))
        OPTIONS_TEXT3 = get_font(30).render("*Tendras que presionar la tecla E del teclado", True, "white")
        OPTIONS_RECT3= OPTIONS_TEXT2.get_rect(center=(400, 330))
        OPTIONS_TEXT4 = get_font(30).render("*El enemigo perseguira de manera tactica y lenta", True, "white")
        OPTIONS_RECT4= OPTIONS_TEXT2.get_rect(center=(400, 390))
        SCREEN.blit(OPTIONS_TEXT, OPTIONS_RECT)
        SCREEN.blit(OPTIONS_TEXT1, OPTIONS_RECT1)
        SCREEN.blit(OPTIONS_TEXT2, OPTIONS_RECT2)
        SCREEN.blit(OPTIONS_TEXT3, OPTIONS_RECT3)
        SCREEN.blit(OPTIONS_TEXT4, OPTIONS_RECT4)

        OPTIONS_BACK = Button(image=None, pos=(400, 650), 
                            text_input="BACK", font=get_font(50), base_color="White", hovering_color="Blue")

        OPTIONS_BACK.changeColor(OPTIONS_MOUSE_POS)
        OPTIONS_BACK.update(SCREEN)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if OPTIONS_BACK.checkForInput(OPTIONS_MOUSE_POS):
                    main_menu()

        pygame.display.update()

def main_menu():
    while True:
        SCREEN.blit(BG, (0, 0)) 

        MENU_MOUSE_POS = pygame.mouse.get_pos()

        MENU_TEXT = get_font(120).render("HOP SQUARE NEW", True, "#000000")
        MENU_RECT = MENU_TEXT.get_rect(center=(400, 100))

        PLAY_BUTTON = Button(image=pygame.image.load("square/Play Rect.png"), pos=(400, 250), 
                             text_input="Play", font=get_font(50), base_color="#d7fcd4", hovering_color="White")
        OPTIONS_BUTTON = Button(image=pygame.image.load("square/Options Rect.png"), pos=(400, 400), 
                            text_input="RulesS", font=get_font(50), base_color="#d7fcd4", hovering_color="White")
        QUIT_BUTTON = Button(image=pygame.image.load("square/Quit Rect.png"), pos=(400, 550), 
                            text_input="QUIT", font=get_font(50), base_color="#d7fcd4", hovering_color="White")

        SCREEN.blit(MENU_TEXT, MENU_RECT)

        for button in [PLAY_BUTTON, OPTIONS_BUTTON, QUIT_BUTTON]:
            button.changeColor(MENU_MOUSE_POS)
            button.update(SCREEN)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if PLAY_BUTTON.checkForInput(MENU_MOUSE_POS):
                    play()
                if OPTIONS_BUTTON.checkForInput(MENU_MOUSE_POS):
                    options()
                if QUIT_BUTTON.checkForInput(MENU_MOUSE_POS):
                    pygame.quit()
                    sys.exit()
        
        pygame.display.update()

def place_powerup_on_platform(platforms):
    """Ubica el power-up en una plataforma aleatoria."""
    platform = random.choice(platforms[1:-1])
    x = random.randint(platform[0], platform[0] + PLATFORM_WIDTH - POWERUP_SIZE)
    y = platform[1] - POWERUP_SIZE
    return PowerUp(x, y)

def generate_platforms():
    """Genera plataformas sin superposición vertical y con espacios saltables."""
    platforms = []
    MIN_HORIZONTAL_DISTANCE = WIDTH // 3
    calculated_max_distance = int(PLAYER_SPEED * (2 * abs(JUMP_STRENGTH) / GRAVITY) ** 0.5)
    MAX_HORIZONTAL_DISTANCE = max(calculated_max_distance, MIN_HORIZONTAL_DISTANCE)
    AVERAGE_VERTICAL_DISTANCE = 1.5 * PLAYER_JUMP
    NUM_PLATFORMS = int(HEIGHT / AVERAGE_VERTICAL_DISTANCE)

    y = HEIGHT - PLAYER_JUMP
    x = random.randint(0, WIDTH - PLATFORM_WIDTH)
    platforms.append((x, y))

    for _ in range(NUM_PLATFORMS):
        y -= random.randint(PLAYER_JUMP, int(1.5 * PLAYER_JUMP))
        direction = random.choice([-1, 1])
        delta_x = random.randint(MIN_HORIZONTAL_DISTANCE, MAX_HORIZONTAL_DISTANCE)
        x += direction * delta_x
        x = max(0, min(WIDTH - PLATFORM_WIDTH, x))
        platforms.append((x, y))

    return platforms

FONT = pygame.font.SysFont(None, 36)

def game_over_menu():
    while True:
        OPTIONS_MOUSE_POS = pygame.mouse.get_pos()

        BACK_BUTTON = Button(image=None, pos=(400, 300), 
                              text_input="RETURN", font=get_font(50), base_color="White", hovering_color="Blue")

        BACK_BUTTON.changeColor(OPTIONS_MOUSE_POS)
        BACK_BUTTON.update(screen)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if BACK_BUTTON.checkForInput(OPTIONS_MOUSE_POS):
                    main_menu()

                    return  # Retorna al bucle principal del juego

        pygame.display.update()


def game_loop():
    global CURRENT_LEVEL
    clock = pygame.time.Clock()
    enemies = []

    while len(enemies) < 1 + CURRENT_LEVEL // 5:
        enemy_x = random.randint(0, WIDTH - ENEMY_SIZE)
        enemy_y = random.randint(0, HEIGHT - ENEMY_SIZE)

        while abs(enemy_x - WIDTH // 2) < SAFE_ZONE and abs(enemy_y - HEIGHT + PLAYER_SIZE + 10) < SAFE_ZONE:
            enemy_x = random.randint(0, WIDTH - ENEMY_SIZE)
            enemy_y = random.randint(0, HEIGHT - ENEMY_SIZE)

        enemy = LearningEnemy()
        enemy.dx += ENEMY_INCREMENT_SPEED * CURRENT_LEVEL
        enemy.dy += ENEMY_INCREMENT_SPEED * CURRENT_LEVEL
        enemies.append(enemy)

    player = Player()
    platforms = generate_platforms()
    powerup = place_powerup_on_platform(platforms)
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_e:
                    last_platform = pygame.Rect(platforms[-1][0], platforms[-1][1], PLATFORM_WIDTH, PLATFORM_HEIGHT)
                    player_rect = pygame.Rect(player.x - 5, player.y - 5, PLAYER_SIZE + 10, PLAYER_SIZE + 10)

                    if player_rect.colliderect(last_platform):
                        CURRENT_LEVEL += 1
                        game_loop()

        player.move(platforms)
        player.check_powerup_duration()
        if powerup:
            powerup_rect = pygame.Rect(powerup.x, powerup.y, POWERUP_SIZE, POWERUP_SIZE)
        player_rect = pygame.Rect(player.x, player.y, PLAYER_SIZE, PLAYER_SIZE)

        if powerup:
            powerup_rect = pygame.Rect(powerup.x, powerup.y, POWERUP_SIZE, POWERUP_SIZE)
            if player_rect.colliderect(powerup_rect) and not player.powered_up:
                player.activate_powerup()
                powerup = None

        for enemy in enemies:
            enemy.move(player)

            enemy_rect = pygame.Rect(enemy.x, enemy.y, ENEMY_SIZE, ENEMY_SIZE)
            player_rect = pygame.Rect(player.x, player.y, PLAYER_SIZE, PLAYER_SIZE)

            if player_rect.colliderect(enemy_rect):
                if player.powered_up:
                    enemies.remove(enemy)
                else:
                    player.times_hit += 1
                    if player.times_hit == 3:
                        CURRENT_LEVEL = 1
                        game_over_menu()

        screen.blit(background_image, (0, 0))
        
        if powerup:
            powerup.draw(screen)
        player.draw(screen)
        for enemy in enemies:
            enemy.draw(screen)

        LAST_PLATFORM_IMAGE = pygame.image.load("square/imagen_last_platform.png")  
        PLATFORM_IMAGE = pygame.image.load("square/imagen_platform.png") 

        for i, platform in enumerate(platforms):
            if i == len(platforms) - 1:
                image = LAST_PLATFORM_IMAGE
            else:
                image = PLATFORM_IMAGE

            screen.blit(image, (platform[0], platform[1]))

        font = pygame.font.SysFont(None, 36)
        level_text = font.render(f'Level: {CURRENT_LEVEL}', True, (255, 255, 255))
        screen.blit(level_text, (10, 10))

        pygame.display.flip()

        clock.tick(60)

    pygame.quit()


def run_game():
    while True:
        choice = main_menu()

        if choice == "PLAY":
            # Llamar a la función principal del juego
            game_loop()
        elif choice == "RULES":
            options()
        elif choice == "QUIT":
            pygame.quit()
            sys.exit()

if __name__ == "__main__":
    #pygame.init()
    run_game()