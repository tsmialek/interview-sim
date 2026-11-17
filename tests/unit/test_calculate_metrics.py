import jiwer

from calculate_metrics import (
    parse_reference,
    normalize_for_cer_wer,
    normalize_for_terms,
)


def test_calculate_metrics():
    out = jiwer.process_words("ala ma kota i psa", "ala ma kot i psa i")
    print(out)

    assert 1 == 0


def test_parse_reference_returns_text_with_terms_list():
    transcript = " Yyy prywatnie korzystałem z {AWS|AWS} z systemów {EC2|EC2} oraz yyy z yyy funkcji klaudo- yyy z {Cloud functions|Cloud Functions}. Eee i próbowałem też {Self-hostować|self-hosting} {PocketBase|PocketBase}"
    text, terms_list = parse_reference(transcript)
    assert (
        text
        == "Yyy prywatnie korzystałem z AWS z systemów EC2 oraz yyy z yyy funkcji klaudo- yyy z Cloud functions. Eee i próbowałem też Self-hostować PocketBase"
    )
    assert terms_list == [
        ("AWS", "AWS"),
        ("EC2", "EC2"),
        ("Cloud functions", "Cloud Functions"),
        ("Self-hostować", "self-hosting"),
        ("PocketBase", "PocketBase"),
    ]


def test_normalize_for_cer_wer():
    transcript = "Yyy prywatnie mmm korzystałem y z AWS eee z systemów EC2 oraz yyy z yyy funkcji klaudo- yyy z Cloud functions. Eee i próbowałem też Self-hostować PocketBase"
    result = normalize_for_cer_wer(transcript)
    assert (
        result
        == "prywatnie korzystałem z aws z systemów ec2 oraz z funkcji z cloud functions i próbowałem też selfhostować pocketbase"
    )


def test_normalize_for_terms():
    transcript = "Yyy jest to aplikacja której 'backend' jest zrobiony w C# przy pomocy yyy \"Frameworku\" (.NET) yyy a Frontend jest... zrobiony w Next.js. yyy całość Hostujemy na Railwayu. Backend yyy i bazę danych która jest bazą yyy w Postgresie yyy wektor- do wektoryzacji"
    result = normalize_for_terms(transcript)
    assert (
        result
        == "yyy jest to aplikacja której backend jest zrobiony w c# przy pomocy yyy frameworku .net yyy a frontend jest zrobiony w next.js yyy całość hostujemy na railwayu backend yyy i bazę danych która jest bazą yyy w postgresie yyy wektor- do wektoryzacji"
    )
