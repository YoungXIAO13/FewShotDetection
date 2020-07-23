#!/usr/bin/env bash

wget --header 'Host: uc876c05c27e6ed81d264c7757b7.dl.dropboxusercontent.com' --user-agent 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:66.0) Gecko/20100101 Firefox/66.0' --header 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' --header 'Accept-Language: en-US,en;q=0.5' --referer 'https://www.dropbox.com/' --header 'Upgrade-Insecure-Requests: 1' 'https://uc876c05c27e6ed81d264c7757b7.dl.dropboxusercontent.com/cd/0/get/A8Fn_TfDvAq_kF5_reg0Do_77AuvnBcSPqRArDOX8MA6_flwb8NxJkk9tlA2BLF-SvidENHUZCICRX5oTYQQqpzrf8s2YR_gvh7fEanTQBXxCz46Db1hCb0jULkLgxZTyWc/file#' --output-document 'FewShotDetectionBaseModels.zip'

unzip FewShotDetectionBaseModels.zip && rm FewShotDetectionBaseModels.zip