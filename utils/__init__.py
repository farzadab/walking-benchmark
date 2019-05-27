def infrange(start=0, step=1):
    i = start
    while True:
        yield i
        i += step
