from ger_wiki.reader import GerReader


class TestSpaceevalConllReader:
    def test_read_from_file(self):
        reader = GerReader()
        data_path = "tests/fixtures/reader/toy_data.txt"
        instances = reader.read(data_path)

        assert len(instances) == 2

        fields = instances[0].fields
        expected_tokens = [
            "Bunchrew",
            "had",
            "a",
            "station",
            "on",
            "the",
            "Inverness",
            "and",
            "Ross",
            "-",
            "shire",
            "Railway",
            ".",
        ]
        expected_tags = (
            "U-PLACE_NAM",
            "O",
            "O",
            "O",
            "O",
            "O",
            "B-PLACE_NAM",
            "I-PLACE_NAM",
            "I-PLACE_NAM",
            "I-PLACE_NAM",
            "I-PLACE_NAM",
            "L-PLACE_NAM",
            "O",
        )

        assert [t.text for t in fields["tokens"].tokens] == expected_tokens
        assert fields["tags"].labels == list(expected_tags)
