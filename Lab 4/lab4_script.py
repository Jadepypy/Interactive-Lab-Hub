def display(image):
    '''display <image> on piscreen'''


def oled(hp):
    '''show <hp>, maybe with a red bar or whatever is convenient to you'''


def joystick():
    '''read x, y as the Lab4 example. Return True if |x-512| + |y-512| >= 200 (or other threshold, to be determined). False if not'''


def prox_sensor():
    '''read prox as the Lab4 example. Return True if prox <= 100 (or other threshold, to be determined). False if not'''


def light_sensor():
    '''read dark as the Lab4 example. Return True if dark >= 225 (or other threshold, to be determined). False if not'''


def rotary():
    '''read rot as the Lab4 example. return True if rot >= 12 (or other threshold, to be determined). False if not'''


'''there will be many initializing codes here'''
# initialize capacity twizzer
i2c = busio.I2C(board.SCL, board.SDA)
mpr121 = adafruit_mpr121.MPR121(i2c)
# max hp here
hp = 5
while True:
    oled(hp)

    # 0 = lightning, 1= fireball, 2 = earthquake, 3 = slime
    if mpr121[0].value:
        display("lightning.jpg")
        start_time = time.time()
        trigger = False
        while time.time() - start_time < 2:
            if joystick() or prox_sensor() or rotary():
                trigger = False
                break
            if light_sensor():
                trigger = True
                break
    if mpr121[1].value:
        display("fireball.jpg")
        start_time = time.time()
        trigger = False
        while time.time() - start_time < 2:
            if light_sensor() or prox_sensor() or rotary():
                trigger = False
                break
            if joystick():
                trigger = True
                break
    if mpr121[2].value:
        display("earthquake.jpg")
        start_time = time.time()
        trigger = False
        while time.time() - start_time < 2:
            if joystick() or light_sensor() or rotary():
                trigger = False
                break
            if prox_sensor():
                trigger = True
                break
    if mpr121[3].value:
        display("slime.jpg")
        start_time = time.time()
        trigger = False
        while time.time() - start_time < 2:
            if joystick() or light_sensor() or prox_sensor():
                trigger = False
                break
            if rotary():
                trigger = True
                break

        if not trigger:
            hp -= 1

        if hp == 0:
            display("skeleton.jpg")