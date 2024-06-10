## Introduction

fastText-jni is a Java wrapper for [fastText](https://github.com/facebookresearch/fastText) inference.

## Usage

`implementation group: 'com.diffbot', name: 'fasttext-jni', version: '0.9.2.8'`

```java
FastTextModel model;
try(InputStream inputStream = this.getClass().getResourceAsStream(MODEL)) {
    model = new FastTextModel(inputStream);
}
Prediction prediction = model.predictProba(TEXT);
System.out.println(prediction.label + " : " + prediction.probability);
model.close();
```

## Development

### Install git lfs

* Install git lfs
  * Mac OSX: `brew install git-lfs`
  * Archlinux: `pacman -S git-lfs`
  * Other: https://github.com/git-lfs/git-lfs/wiki/Installation
* `git lfs install` in the git repo

`git submodule update --init`