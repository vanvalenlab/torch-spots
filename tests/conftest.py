import pytest

from tests.oracle import load_oracle


@pytest.fixture(scope="session")
def oracle():
    return load_oracle()
