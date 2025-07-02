[
    {
        class_name: 'lexsubgen.post_processors.base_postprocessor.LowercasePostProcessor'
    },
    {
        class_name: "lexsubgen.post_processors.lemmatizer.Lemmatizer",
        lemmatizer: "nltk",
        strategy: "max"
    },
    {
        class_name: "lexsubgen.post_processors.target_excluder.TargetExcluder",
        lemmatizer: "spacy_old"
    }
]
