def make_bricks(small_count, big_count, goal):
    big_parts = goal // 5
    print(f"{big_parts}")
    if big_parts <= big_count:
        minus_value = goal - (big_parts *5)
        if minus_value <= small_count:
            return True
    return False


# small_count*x + big_count*y = goal

print(make_bricks(3, 1, 8)) #  [3x1+1x5 = 8]
print(make_bricks(3, 1, 9))
print(make_bricks(3, 2, 10))
print(make_bricks(3, 2, 11))
