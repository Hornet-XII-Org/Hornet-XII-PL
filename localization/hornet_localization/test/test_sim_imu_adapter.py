import pytest
from geometry_msgs.msg import Vector3

from hornet_localization.sim_imu_adapter_node import flip_to_base_link


def test_flips_upside_down_simulator_imu_into_base_link():
    vector = Vector3(x=1.0, y=2.0, z=-9.81)

    flip_to_base_link(vector)

    assert (vector.x, vector.y, vector.z) == pytest.approx(
        (1.0, -2.0, 9.81)
    )
