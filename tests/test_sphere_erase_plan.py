from sim2chip.photo2scene.sphere import erase_plan


def _obj(lon, lat, width, height, area=0.001):
    return {"lon": lon, "lat": lat, "width": width, "height": height, "area_share": area}


def test_neighbours_share_a_view_and_far_things_do_not():
    cars = [_obj(10, -5, 20, 8, 0.004), _obj(28, -5, 15, 7), _obj(170, -3, 10, 5)]
    groups = erase_plan.group(cars)
    assert sorted(len(g) for g in groups) == [1, 2]


def test_a_group_wider_than_a_view_is_split():
    row = [_obj(lon, -5, 25, 8) for lon in range(0, 200, 22)]  # a row of parked cars, side by side
    for g in erase_plan.group(row):
        box = erase_plan.union_box(g)
        assert box["width"] + 2 * erase_plan.MARGIN_DEG <= erase_plan.MAX_FOV_DEG


def test_the_view_is_centred_across_the_seam_and_fits():
    pair = [_obj(175, -10, 10, 6), _obj(-175, -10, 10, 6)]  # either side of the panorama's seam
    box = erase_plan.union_box(pair)
    assert abs(box["width"] - 20) < 1e-6
    v = erase_plan.view_for(pair, 4096 / 360)
    assert abs((v["yaw_deg"] - 180 + 180) % 360 - 180) < 1e-6 and v["pitch_deg"] == 10
    assert v["fov_deg"] >= erase_plan.MIN_FOV_DEG and v["size"][0] % 16 == 0 and v["size"][0] <= erase_plan.MAX_SIDE
