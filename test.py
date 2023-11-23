
def f(temp_limit=30):
    global temperature, leds_top 
    if temperature > temp_limit*10:
        leds_top = [32,0,0]
    else:
        leds_top= [0,10,32]

