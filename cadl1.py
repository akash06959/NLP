{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNsSMNSyvWfZETO/uqhdbdW",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/akash06959/NLP/blob/main/cadl1.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 755
        },
        "id": "DjZamn_EBV1R",
        "outputId": "fa28930c-e976-4efe-f786-e631a014fd6d"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Original Text:\n",
            " Apple is looking at buying U.K. startup for $1 billion. The company has big plans for AI.\n"
          ]
        },
        {
          "output_type": "error",
          "ename": "LookupError",
          "evalue": "\n**********************************************************************\n  Resource \u001b[93mpunkt_tab\u001b[0m not found.\n  Please use the NLTK Downloader to obtain the resource:\n\n  \u001b[31m>>> import nltk\n  >>> nltk.download('punkt_tab')\n  \u001b[0m\n  For more information see: https://www.nltk.org/data.html\n\n  Attempted to load \u001b[93mtokenizers/punkt_tab/english/\u001b[0m\n\n  Searched in:\n    - '/root/nltk_data'\n    - '/usr/nltk_data'\n    - '/usr/share/nltk_data'\n    - '/usr/lib/nltk_data'\n    - '/usr/share/nltk_data'\n    - '/usr/local/share/nltk_data'\n    - '/usr/lib/nltk_data'\n    - '/usr/local/lib/nltk_data'\n**********************************************************************\n",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mLookupError\u001b[0m                               Traceback (most recent call last)",
            "\u001b[0;32m/tmp/ipython-input-2569831403.py\u001b[0m in \u001b[0;36m<cell line: 0>\u001b[0;34m()\u001b[0m\n\u001b[1;32m     16\u001b[0m \u001b[0;31m# 4. Tokenization\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     17\u001b[0m \u001b[0;31m# -------------------------------\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 18\u001b[0;31m \u001b[0mtokens\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mword_tokenize\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtext\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     19\u001b[0m \u001b[0mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"\\n--- Tokenization ---\\n\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtokens\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     20\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.12/dist-packages/nltk/tokenize/__init__.py\u001b[0m in \u001b[0;36mword_tokenize\u001b[0;34m(text, language, preserve_line)\u001b[0m\n\u001b[1;32m    140\u001b[0m     \u001b[0;34m:\u001b[0m\u001b[0mtype\u001b[0m \u001b[0mpreserve_line\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mbool\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    141\u001b[0m     \"\"\"\n\u001b[0;32m--> 142\u001b[0;31m     \u001b[0msentences\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m[\u001b[0m\u001b[0mtext\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;32mif\u001b[0m \u001b[0mpreserve_line\u001b[0m \u001b[0;32melse\u001b[0m \u001b[0msent_tokenize\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtext\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlanguage\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    143\u001b[0m     return [\n\u001b[1;32m    144\u001b[0m         \u001b[0mtoken\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0msent\u001b[0m \u001b[0;32min\u001b[0m \u001b[0msentences\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0mtoken\u001b[0m \u001b[0;32min\u001b[0m \u001b[0m_treebank_word_tokenizer\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtokenize\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0msent\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.12/dist-packages/nltk/tokenize/__init__.py\u001b[0m in \u001b[0;36msent_tokenize\u001b[0;34m(text, language)\u001b[0m\n\u001b[1;32m    117\u001b[0m     \u001b[0;34m:\u001b[0m\u001b[0mparam\u001b[0m \u001b[0mlanguage\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mthe\u001b[0m \u001b[0mmodel\u001b[0m \u001b[0mname\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mthe\u001b[0m \u001b[0mPunkt\u001b[0m \u001b[0mcorpus\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    118\u001b[0m     \"\"\"\n\u001b[0;32m--> 119\u001b[0;31m     \u001b[0mtokenizer\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0m_get_punkt_tokenizer\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mlanguage\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    120\u001b[0m     \u001b[0;32mreturn\u001b[0m \u001b[0mtokenizer\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtokenize\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtext\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    121\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.12/dist-packages/nltk/tokenize/__init__.py\u001b[0m in \u001b[0;36m_get_punkt_tokenizer\u001b[0;34m(language)\u001b[0m\n\u001b[1;32m    103\u001b[0m     \u001b[0;34m:\u001b[0m\u001b[0mtype\u001b[0m \u001b[0mlanguage\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mstr\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    104\u001b[0m     \"\"\"\n\u001b[0;32m--> 105\u001b[0;31m     \u001b[0;32mreturn\u001b[0m \u001b[0mPunktTokenizer\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mlanguage\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    106\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    107\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.12/dist-packages/nltk/tokenize/punkt.py\u001b[0m in \u001b[0;36m__init__\u001b[0;34m(self, lang)\u001b[0m\n\u001b[1;32m   1742\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m__init__\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlang\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m\"english\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1743\u001b[0m         \u001b[0mPunktSentenceTokenizer\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m__init__\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1744\u001b[0;31m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mload_lang\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mlang\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1745\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1746\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mload_lang\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlang\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m\"english\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.12/dist-packages/nltk/tokenize/punkt.py\u001b[0m in \u001b[0;36mload_lang\u001b[0;34m(self, lang)\u001b[0m\n\u001b[1;32m   1747\u001b[0m         \u001b[0;32mfrom\u001b[0m \u001b[0mnltk\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdata\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mfind\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1748\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1749\u001b[0;31m         \u001b[0mlang_dir\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mfind\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34mf\"tokenizers/punkt_tab/{lang}/\"\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1750\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_params\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mload_punkt_params\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mlang_dir\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1751\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_lang\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mlang\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.12/dist-packages/nltk/data.py\u001b[0m in \u001b[0;36mfind\u001b[0;34m(resource_name, paths)\u001b[0m\n\u001b[1;32m    577\u001b[0m     \u001b[0msep\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m\"*\"\u001b[0m \u001b[0;34m*\u001b[0m \u001b[0;36m70\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    578\u001b[0m     \u001b[0mresource_not_found\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34mf\"\\n{sep}\\n{msg}\\n{sep}\\n\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 579\u001b[0;31m     \u001b[0;32mraise\u001b[0m \u001b[0mLookupError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mresource_not_found\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    580\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    581\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mLookupError\u001b[0m: \n**********************************************************************\n  Resource \u001b[93mpunkt_tab\u001b[0m not found.\n  Please use the NLTK Downloader to obtain the resource:\n\n  \u001b[31m>>> import nltk\n  >>> nltk.download('punkt_tab')\n  \u001b[0m\n  For more information see: https://www.nltk.org/data.html\n\n  Attempted to load \u001b[93mtokenizers/punkt_tab/english/\u001b[0m\n\n  Searched in:\n    - '/root/nltk_data'\n    - '/usr/nltk_data'\n    - '/usr/share/nltk_data'\n    - '/usr/lib/nltk_data'\n    - '/usr/share/nltk_data'\n    - '/usr/local/share/nltk_data'\n    - '/usr/lib/nltk_data'\n    - '/usr/local/lib/nltk_data'\n**********************************************************************\n"
          ]
        }
      ],
      "source": [
        "# ===============================\n",
        "# CADL1 – Text Preprocessing\n",
        "import spacy\n",
        "from nltk.tokenize import word_tokenize\n",
        "from nltk.corpus import stopwords\n",
        "from nltk.stem import PorterStemmer, WordNetLemmatizer\n",
        "\n",
        "# -------------------------------\n",
        "# 3. Sample Text\n",
        "# -------------------------------\n",
        "text = \"Apple is looking at buying U.K. startup for $1 billion. The company has big plans for AI.\"\n",
        "\n",
        "print(\"Original Text:\\n\", text)\n",
        "\n",
        "# -------------------------------\n",
        "# 4. Tokenization\n",
        "# -------------------------------\n",
        "tokens = word_tokenize(text)\n",
        "print(\"\\n--- Tokenization ---\\n\", tokens)\n",
        "\n",
        "# -------------------------------\n",
        "# 5. Stopword Removal\n",
        "# -------------------------------\n",
        "stop_words = set(stopwords.words('english'))\n",
        "filtered_tokens = [w for w in tokens if w.lower() not in stop_words]\n",
        "print(\"\\n--- After Stopword Removal ---\\n\", filtered_tokens)\n",
        "\n",
        "# -------------------------------\n",
        "# 6. Stemming\n",
        "# -------------------------------\n",
        "ps = PorterStemmer()\n",
        "stemmed = [ps.stem(w) for w in filtered_tokens]\n",
        "print(\"\\n--- After Stemming ---\\n\", stemmed)\n",
        "\n",
        "# -------------------------------\n",
        "# 7. Lemmatization (WordNet)\n",
        "# -------------------------------\n",
        "lemmatizer = WordNetLemmatizer()\n",
        "lemmatized = [lemmatizer.lemmatize(w) for w in filtered_tokens]\n",
        "print(\"\\n--- After Lemmatization (WordNet) ---\\n\", lemmatized)\n",
        "\n",
        "# -------------------------------\n",
        "# 8. Lemmatization (spaCy)\n",
        "# -------------------------------\n",
        "nlp = spacy.load(\"en_core_web_sm\")\n",
        "doc = nlp(text)\n",
        "print(\"\\n--- Lemmatization (spaCy) ---\")\n",
        "for token in doc:\n",
        "    print(f\"{token.text:15} → {token.lemma_}\")\n"
      ]
    }
  ]
}