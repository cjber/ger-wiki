from allennlp.common.testing import ModelTestCase
from allennlp_models.tagging.models import CrfTagger


class TestSimpleClassifier(ModelTestCase):
    def test_model_can_train(self):
        param_file = "tests/fixtures/model/test_config.jsonnet"
        self.ensure_model_can_train_save_and_load(param_file)
