normalized_passages = ["hello this is a sentence", "i walked my dog", "this is also a sentence"]
normalized_answers = ["hello", "dog", "this is", "DOESNOTEXIST"]
print([answer in passage for answer in normalized_answers for passage in normalized_passages])
print(any([answer in passage for answer in normalized_answers for passage in normalized_passages]))
normalized_answers = ["hello", "dog", "DOESNOTEXIST"]
print([answer in passage for answer in normalized_answers for passage in normalized_passages])
print(any([answer in passage for answer in normalized_answers for passage in normalized_passages]))
normalized_answers = ["hello", "DOESNOTEXIST"]
print([answer in passage for answer in normalized_answers for passage in normalized_passages])
print(any([answer in passage for answer in normalized_answers for passage in normalized_passages]))
normalized_answers = ["DOESNOTEXIST"]
print([answer in passage for answer in normalized_answers for passage in normalized_passages])
print(any([answer in passage for answer in normalized_answers for passage in normalized_passages]))
