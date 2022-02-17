## 如何在HyperGBM中自定义存储空间？

HyperGBM 在运行时将中间数据（缓存数据、模型文件等）存储在系 *work_dir* 中, 默认情况下  *work_dir*时系统临时目录下的一个子目录。

---

* 设置 *work_dir* 位置

  您可以通过修改系统临时目录来调整 *work_dir* 的位置，另外您也可以通过如下方式设置*work_dir*的位置

1. 在您运行HyerGBM的工作目录中创建一个子目录 `conf` 

2. 在*conf* 目录下创建文件 `storage.py` ，在其中指定 *work_dir* 的位置，如下:
   ```
   c.StorageCfg.root = '/your/full/path/for/work_dir'
   ```
   
3. 按照常规方式使用HyperGBM

   

---

* 使用兼容 s3 的对象存储作为  *work_dir*
1. 安装`s3fs`

2. 在您运行HyerGBM的工作目录中创建一个子目录 `conf` 

3. 在*conf* 目录下创建文件 `storage.py` ，在其中配置s3存储的信息，示例如下:
   ```
   c.StorageCfg.kind = 's3'
   c.StorageCfg.root = '/bucket_name/some_path'
   c.StorageCfg.options = """
   {
           "anon": false,
           "client_kwargs": {
                   "endpoint_url": "your_service_address",
                   "aws_access_key_id": "your_access_key",
                   "aws_secret_access_key": "your_secret_access_key"
           }
   }
   """
   ```
   
   关于s3存储的详细配置信息请参考 s3fs 的 [官方文档](https://s3fs.readthedocs.io/en/latest/) 。
