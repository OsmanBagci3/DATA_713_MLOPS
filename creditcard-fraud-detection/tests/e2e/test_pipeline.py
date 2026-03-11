"""End-to-end tests (require docker-compose up)."""
import pytest

class TestFullPipeline:
    @pytest.mark.skip(reason="Requires running services")
    def test_data_pipeline(self): pass

    @pytest.mark.skip(reason="Requires running services")
    def test_train_and_serve(self): pass
