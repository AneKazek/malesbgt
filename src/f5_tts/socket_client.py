704.7s	1290	Traceback (most recent call last):
704.7s	1291	  File "/kaggle/working/.venv/lib/python3.11/site-packages/torch/serialization.py", line 944, in save
704.7s	1292	    _save(
704.7s	1293	  File "/kaggle/working/.venv/lib/python3.11/site-packages/torch/serialization.py", line 1216, in _save
704.7s	1294	    zip_file.write_record(name, storage, num_bytes)
704.7s	1295	RuntimeError: [enforce fail at inline_container.cc:815] . PytorchStreamWriter failed writing file data/1661: file write failed
704.7s	1296	
704.7s	1297	During handling of the above exception, another exception occurred:
704.7s	1298	
704.7s	1299	Traceback (most recent call last):
704.7s	1300	  File "/kaggle/working/kcv-tts/src/f5_tts/train/train.py", line 83, in <module>
704.7s	1301	    main()
704.7s	1302	  File "/kaggle/working/.venv/lib/python3.11/site-packages/hydra/main.py", line 94, in decorated_main
704.7s	1303	    _run_hydra(
704.7s	1304	  File "/kaggle/working/.venv/lib/python3.11/site-packages/hydra/_internal/utils.py", line 394, in _run_hydra
704.7s	1305	    _run_app(
704.7s	1306	  File "/kaggle/working/.venv/lib/python3.11/site-packages/hydra/_internal/utils.py", line 457, in _run_app
704.7s	1307	    run_and_report(
704.7s	1308	  File "/kaggle/working/.venv/lib/python3.11/site-packages/hydra/_internal/utils.py", line 223, in run_and_report
704.7s	1309	    raise ex
704.7s	1310	  File "/kaggle/working/.venv/lib/python3.11/site-packages/hydra/_internal/utils.py", line 220, in run_and_report
704.7s	1311	    return func()
704.7s	1312	           ^^^^^^
704.7s	1313	  File "/kaggle/working/.venv/lib/python3.11/site-packages/hydra/_internal/utils.py", line 458, in <lambda>
704.7s	1314	    lambda: hydra.run(
704.7s	1315	            ^^^^^^^^^^
704.7s	1316	  File "/kaggle/working/.venv/lib/python3.11/site-packages/hydra/_internal/hydra.py", line 132, in run
704.7s	1317	    _ = ret.return_value
704.7s	1318	        ^^^^^^^^^^^^^^^^
704.7s	1319	  File "/kaggle/working/.venv/lib/python3.11/site-packages/hydra/core/utils.py", line 260, in return_value
704.7s	1320	    raise self._return_value
704.7s	1321	  File "/kaggle/working/.venv/lib/python3.11/site-packages/hydra/core/utils.py", line 186, in run_job
704.7s	1322	    ret.return_value = task_function(task_cfg)
704.7s	1323	                       ^^^^^^^^^^^^^^^^^^^^^^^
704.7s	1324	  File "/kaggle/working/kcv-tts/src/f5_tts/train/train.py", line 75, in main
704.7s	1325	    trainer.train(
704.7s	1326	  File "/kaggle/working/kcv-tts/src/f5_tts/model/trainer.py", line 489, in train
704.7s	1327	    self.save_checkpoint(global_update)
704.7s	1328	  File "/kaggle/working/kcv-tts/src/f5_tts/model/trainer.py", line 204, in save_checkpoint
704.7s	1329	    self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_{update}.pt")
704.7s	1330	  File "/kaggle/working/.venv/lib/python3.11/site-packages/accelerate/accelerator.py", line 3415, in save
704.7s	1331	    save(
704.7s	1332	  File "/kaggle/working/.venv/lib/python3.11/site-packages/accelerate/utils/other.py", line 387, in save
704.7s	1333	    save_func(obj, f)
704.7s	1334	  File "/kaggle/working/.venv/lib/python3.11/site-packages/torch/serialization.py", line 943, in save
704.7s	1335	    with _open_zipfile_writer(f) as opened_zipfile:
704.7s	1336	  File "/kaggle/working/.venv/lib/python3.11/site-packages/torch/serialization.py", line 784, in __exit__
704.7s	1337	    self.file_like.write_end_of_file()
704.7s	1338	RuntimeError: [enforce fail at inline_container.cc:626] . unexpected pos 2221367680 vs 2221367568
704.7s	1339	[rank0]: Traceback (most recent call last):
704.7s	1340	[rank0]:   File "/kaggle/working/.venv/lib/python3.11/site-packages/torch/serialization.py", line 944, in save
704.7s	1341	[rank0]:     _save(
704.7s	1342	[rank0]:   File "/kaggle/working/.venv/lib/python3.11/site-packages/torch/serialization.py", line 1216, in _save
704.7s	1343	[rank0]:     zip_file.write_record(name, storage, num_bytes)
704.7s	1344	[rank0]: RuntimeError: [enforce fail at inline_container.cc:815] . PytorchStreamWriter failed writing file data/1661: file write failed
704.7s	1345	
704.7s	1346	[rank0]: During handling of the above exception, another exception occurred:
704.7s	1347	
704.7s	1348	[rank0]: Traceback (most recent call last):
704.7s	1349	[rank0]:   File "/kaggle/working/kcv-tts/src/f5_tts/train/train.py", line 83, in <module>
704.7s	1350	[rank0]:     main()
704.7s	1351	[rank0]:   File "/kaggle/working/.venv/lib/python3.11/site-packages/hydra/main.py", line 94, in decorated_main
704.7s	1352	[rank0]:     _run_hydra(
704.7s	1353	[rank0]:   File "/kaggle/working/.venv/lib/python3.11/site-packages/hydra/_internal/utils.py", line 394, in _run_hydra
704.7s	1354	[rank0]:     _run_app(
704.7s	1355	[rank0]:   File "/kaggle/working/.venv/lib/python3.11/site-packages/hydra/_internal/utils.py", line 457, in _run_app
704.7s	1356	[rank0]:     run_and_report(
704.7s	1357	[rank0]:   File "/kaggle/working/.venv/lib/python3.11/site-packages/hydra/_internal/utils.py", line 223, in run_and_report
704.7s	1358	[rank0]:     raise ex
704.7s	1359	[rank0]:   File "/kaggle/working/.venv/lib/python3.11/site-packages/hydra/_internal/utils.py", line 220, in run_and_report
704.7s	1360	[rank0]:     return func()
704.7s	1361	[rank0]:            ^^^^^^
704.7s	1362	[rank0]:   File "/kaggle/working/.venv/lib/python3.11/site-packages/hydra/_internal/utils.py", line 458, in <lambda>
704.7s	1363	[rank0]:     lambda: hydra.run(
704.7s	1364	[rank0]:             ^^^^^^^^^^
704.7s	1365	[rank0]:   File "/kaggle/working/.venv/lib/python3.11/site-packages/hydra/_internal/hydra.py", line 132, in run
704.7s	1366	[rank0]:     _ = ret.return_value
704.7s	1367	[rank0]:         ^^^^^^^^^^^^^^^^
704.7s	1368	[rank0]:   File "/kaggle/working/.venv/lib/python3.11/site-packages/hydra/core/utils.py", line 260, in return_value
704.7s	1369	[rank0]:     raise self._return_value
704.7s	1370	[rank0]:   File "/kaggle/working/.venv/lib/python3.11/site-packages/hydra/core/utils.py", line 186, in run_job
704.7s	1371	[rank0]:     ret.return_value = task_function(task_cfg)
704.7s	1372	[rank0]:                        ^^^^^^^^^^^^^^^^^^^^^^^
704.7s	1373	[rank0]:   File "/kaggle/working/kcv-tts/src/f5_tts/train/train.py", line 75, in main
704.7s	1374	[rank0]:     trainer.train(
704.7s	1375	[rank0]:   File "/kaggle/working/kcv-tts/src/f5_tts/model/trainer.py", line 489, in train
704.7s	1376	[rank0]:     self.save_checkpoint(global_update)
704.7s	1377	[rank0]:   File "/kaggle/working/kcv-tts/src/f5_tts/model/trainer.py", line 204, in save_checkpoint
704.7s	1378	[rank0]:     self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_{update}.pt")
704.7s	1379	[rank0]:   File "/kaggle/working/.venv/lib/python3.11/site-packages/accelerate/accelerator.py", line 3415, in save
704.7s	1380	[rank0]:     save(
704.7s	1381	[rank0]:   File "/kaggle/working/.venv/lib/python3.11/site-packages/accelerate/utils/other.py", line 387, in save
704.7s	1382	[rank0]:     save_func(obj, f)
704.7s	1383	[rank0]:   File "/kaggle/working/.venv/lib/python3.11/site-packages/torch/serialization.py", line 943, in save
704.7s	1384	[rank0]:     with _open_zipfile_writer(f) as opened_zipfile:
704.7s	1385	[rank0]:   File "/kaggle/working/.venv/lib/python3.11/site-packages/torch/serialization.py", line 784, in __exit__
704.7s	1386	[rank0]:     self.file_like.write_end_of_file()
704.7s	1387	[rank0]: RuntimeError: [enforce fail at inline_container.cc:626] . unexpected pos 2221367680 vs 2221367568
704.7s	1388	[rank1]:[E325 17:13:23.163175227 ProcessGroupGloo.cpp:145] Rank 1 successfully reached monitoredBarrier, but received errors while waiting for send/recv from rank 0. Please check rank 0 logs for faulty rank.
704.7s	1389	Error executing job with overrides: ['++datasets.name=datasetku', '++datasets.batch_size_per_gpu=512', '++datasets.batch_size_type=frame', '++datasets.max_samples=2', '++datasets.num_workers=4', '++optim.epochs=30', '++optim.learning_rate=1e-05', '++optim.num_warmup_updates=200', '++optim.grad_accumulation_steps=4', '++optim.max_grad_norm=1.0', '++optim.bnb_optimizer=False', '++ckpts.log_samples=False', '++ckpts.save_per_updates=100', '++ckpts.last_per_updates=50', '++ckpts.keep_last_n_checkpoints=3', '++model.name=F5TTS_MAMBA_kaggle_full', '++model.tokenizer=pinyin', '++model.arch.use_mamba=false', '++ckpts.logger=wandb', '++ckpts.wandb_project=kcvanguard', '++ckpts.wandb_run_name=F5TTS_MAMBA_kaggle_full']
704.7s	1390	[rank1]: Traceback (most recent call last):
704.7s	1391	[rank1]:   File "/kaggle/working/kcv-tts/src/f5_tts/train/train.py", line 83, in <module>
704.7s	1392	[rank1]:     main()
704.7s	1393	[rank1]:   File "/kaggle/working/.venv/lib/python3.11/site-packages/hydra/main.py", line 94, in decorated_main
704.7s	1394	[rank1]:     _run_hydra(
704.7s	1395	[rank1]:   File "/kaggle/working/.venv/lib/python3.11/site-packages/hydra/_internal/utils.py", line 394, in _run_hydra
704.7s	1396	[rank1]:     _run_app(
704.7s	1397	[rank1]:   File "/kaggle/working/.venv/lib/python3.11/site-packages/hydra/_internal/utils.py", line 457, in _run_app
704.7s	1398	[rank1]:     run_and_report(
704.7s	1399	[rank1]:   File "/kaggle/working/.venv/lib/python3.11/site-packages/hydra/_internal/utils.py", line 223, in run_and_report
704.7s	1400	[rank1]:     raise ex
704.7s	1401	[rank1]:   File "/kaggle/working/.venv/lib/python3.11/site-packages/hydra/_internal/utils.py", line 220, in run_and_report
704.7s	1402	[rank1]:     return func()
704.7s	1403	[rank1]:            ^^^^^^
704.7s	1404	[rank1]:   File "/kaggle/working/.venv/lib/python3.11/site-packages/hydra/_internal/utils.py", line 458, in <lambda>
704.7s	1405	[rank1]:     lambda: hydra.run(
704.7s	1406	[rank1]:             ^^^^^^^^^^
704.7s	1407	[rank1]:   File "/kaggle/working/.venv/lib/python3.11/site-packages/hydra/_internal/hydra.py", line 132, in run
704.7s	1408	[rank1]:     _ = ret.return_value
704.7s	1409	[rank1]:         ^^^^^^^^^^^^^^^^
704.7s	1410	[rank1]:   File "/kaggle/working/.venv/lib/python3.11/site-packages/hydra/core/utils.py", line 260, in return_value
704.7s	1411	[rank1]:     raise self._return_value
704.7s	1412	[rank1]:   File "/kaggle/working/.venv/lib/python3.11/site-packages/hydra/core/utils.py", line 186, in run_job
704.7s	1413	[rank1]:     ret.return_value = task_function(task_cfg)
704.7s	1414	[rank1]:                        ^^^^^^^^^^^^^^^^^^^^^^^
704.7s	1415	[rank1]:   File "/kaggle/working/kcv-tts/src/f5_tts/train/train.py", line 75, in main
704.7s	1416	[rank1]:     trainer.train(
704.7s	1417	[rank1]:   File "/kaggle/working/kcv-tts/src/f5_tts/model/trainer.py", line 440, in train
704.7s	1418	[rank1]:     loss, cond, pred = self.model(
704.7s	1419	[rank1]:                        ^^^^^^^^^^^
704.7s	1420	[rank1]:   File "/kaggle/working/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
704.7s	1421	[rank1]:     return self._call_impl(*args, **kwargs)
704.7s	1422	[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
704.7s	1423	[rank1]:   File "/kaggle/working/.venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1750, in _call_impl
704.7s	1424	[rank1]:     return forward_call(*args, **kwargs)
704.7s	1425	[rank1]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
704.7s	1426	[rank1]:   File "/kaggle/working/.venv/lib/python3.11/site-packages/torch/nn/parallel/distributed.py", line 1639, in forward
704.7s	1427	[rank1]:     inputs, kwargs = self._pre_forward(*inputs, **kwargs)
704.7s	1428	[rank1]:                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
704.7s	1429	[rank1]:   File "/kaggle/working/.venv/lib/python3.11/site-packages/torch/nn/parallel/distributed.py", line 1535, in _pre_forward
704.7s	1430	[rank1]:     self._sync_buffers()
704.7s	1431	[rank1]:   File "/kaggle/working/.venv/lib/python3.11/site-packages/torch/nn/parallel/distributed.py", line 2172, in _sync_buffers
704.7s	1432	[rank1]:     self._sync_module_buffers(authoritative_rank)
704.7s	1433	[rank1]:   File "/kaggle/working/.venv/lib/python3.11/site-packages/torch/nn/parallel/distributed.py", line 2176, in _sync_module_buffers
704.7s	1434	[rank1]:     self._default_broadcast_coalesced(authoritative_rank=authoritative_rank)
704.7s	1435	[rank1]:   File "/kaggle/working/.venv/lib/python3.11/site-packages/torch/nn/parallel/distributed.py", line 2198, in _default_broadcast_coalesced
704.7s	1436	[rank1]:     self._distributed_broadcast_coalesced(bufs, bucket_size, authoritative_rank)
704.7s	1437	[rank1]:   File "/kaggle/working/.venv/lib/python3.11/site-packages/torch/nn/parallel/distributed.py", line 2113, in _distributed_broadcast_coalesced
704.7s	1438	[rank1]:     dist._broadcast_coalesced(
704.7s	1439	[rank1]: RuntimeError: Rank 1 successfully reached monitoredBarrier, but received errors while waiting for send/recv from rank 0. Please check rank 0 logs for faulty rank.
704.7s	1440	[rank1]:  Original exception: 
704.7s	1441	[rank1]: [/pytorch/third_party/gloo/gloo/transport/tcp/pair.cc:534] Connection closed by peer [172.19.2.2]:33177
704.7s	1442	W0325 17:13:24.151000 1874 torch/distributed/elastic/multiprocessing/api.py:897] Sending process 1887 closing signal SIGTERM
704.7s	1443	E0325 17:13:24.154000 1874 torch/distributed/elastic/multiprocessing/api.py:869] failed (exitcode: 1) local_rank: 0 (pid: 1886) of binary: /kaggle/working/.venv/bin/python
704.7s	1444	Traceback (most recent call last):
704.7s	1445	  File "/kaggle/working/.venv/bin/accelerate", line 6, in <module>
704.7s	1446	    sys.exit(main())
704.7s	1447	             ^^^^^^
704.7s	1448	  File "/kaggle/working/.venv/lib/python3.11/site-packages/accelerate/commands/accelerate_cli.py", line 50, in main
704.7s	1449	    args.func(args)
704.7s	1450	  File "/kaggle/working/.venv/lib/python3.11/site-packages/accelerate/commands/launch.py", line 1396, in launch_command
704.7s	1451	    multi_gpu_launcher(args)
704.7s	1452	  File "/kaggle/working/.venv/lib/python3.11/site-packages/accelerate/commands/launch.py", line 1023, in multi_gpu_launcher
704.7s	1453	    distrib_run.run(args)
704.7s	1454	  File "/kaggle/working/.venv/lib/python3.11/site-packages/torch/distributed/run.py", line 909, in run
704.7s	1455	    elastic_launch(
704.7s	1456	  File "/kaggle/working/.venv/lib/python3.11/site-packages/torch/distributed/launcher/api.py", line 138, in __call__
704.7s	1457	    return launch_agent(self._config, self._entrypoint, list(args))
704.7s	1458	           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
704.7s	1459	  File "/kaggle/working/.venv/lib/python3.11/site-packages/torch/distributed/launcher/api.py", line 269, in launch_agent
704.7s	1460	    raise ChildFailedError(
704.7s	1461	torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
704.7s	1462	============================================================
704.7s	1463	src/f5_tts/train/train.py FAILED
704.7s	1464	------------------------------------------------------------
704.7s	1465	Failures:
704.7s	1466	  <NO_OTHER_FAILURES>
704.7s	1467	------------------------------------------------------------
704.7s	1468	Root Cause (first observed failure):
704.7s	1469	[0]:
704.7s	1470	  time      : 2026-03-25_17:13:24
704.7s	1471	  host      : 0f0baed86552
704.7s	1472	  rank      : 0 (local_rank: 0)
704.7s	1473	  exitcode  : 1 (pid: 1886)
704.7s	1474	  error_file: <N/A>
704.7s	1475	  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
704.7s	1476	============================================================
705.1s	1477	Traceback (most recent call last):
705.1s	1478	  File "/usr/local/lib/python3.12/dist-packages/papermill/engines.py", line 371, in execute_notebook
705.1s	1479	    cls.execute_managed_notebook(nb_man, kernel_name, log_output=log_output, **kwargs)
705.1s	1480	  File "/usr/local/lib/python3.12/dist-packages/papermill/engines.py", line 443, in execute_managed_notebook
705.1s	1481	    return PapermillNotebookClient(nb_man, **final_kwargs).execute()
705.1s	1482	           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
705.1s	1483	  File "/usr/local/lib/python3.12/dist-packages/papermill/clientwrap.py", line 45, in execute
705.1s	1484	    self.papermill_execute_cells()
705.1s	1485	  File "/usr/local/lib/python3.12/dist-packages/papermill/clientwrap.py", line 77, in papermill_execute_cells
705.1s	1486	    self.nb_man.cell_complete(self.nb.cells[index], cell_index=index)
705.1s	1487	  File "/usr/local/lib/python3.12/dist-packages/papermill/engines.py", line 77, in wrapper
705.1s	1488	    return func(self, *args, **kwargs)
705.1s	1489	           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
705.1s	1490	  File "/usr/local/lib/python3.12/dist-packages/papermill/engines.py", line 274, in cell_complete
705.1s	1491	    self.save()
705.1s	1492	  File "/usr/local/lib/python3.12/dist-packages/papermill/engines.py", line 77, in wrapper
705.1s	1493	    return func(self, *args, **kwargs)
705.1s	1494	           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
705.1s	1495	  File "/usr/local/lib/python3.12/dist-packages/papermill/engines.py", line 153, in save
705.1s	1496	    write_ipynb(self.nb, self.output_path)
705.1s	1497	  File "/usr/local/lib/python3.12/dist-packages/papermill/iorw.py", line 486, in write_ipynb
705.1s	1498	    papermill_io.write(nbformat.writes(nb), path)
705.1s	1499	  File "/usr/local/lib/python3.12/dist-packages/papermill/iorw.py", line 102, in write
705.1s	1500	    return self.get_handler(path, extensions).write(buf, path)
705.1s	1501	           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
705.1s	1502	  File "/usr/local/lib/python3.12/dist-packages/papermill/iorw.py", line 215, in write
705.1s	1503	    f.write(buf)
705.1s	1504	OSError: [Errno 28] No space left on device
705.1s	1505	
705.1s	1506	During handling of the above exception, another exception occurred:
705.1s	1507	
705.1s	1508	Traceback (most recent call last):
705.1s	1509	  File "<string>", line 1, in <module>
705.1s	1510	  File "/usr/local/lib/python3.12/dist-packages/papermill/execute.py", line 116, in execute_notebook
705.1s	1511	    nb = papermill_engines.execute_notebook_with_engine(
705.1s	1512	         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
705.1s	1513	  File "/usr/local/lib/python3.12/dist-packages/papermill/engines.py", line 49, in execute_notebook_with_engine
705.1s	1514	    return self.get_engine(engine_name).execute_notebook(nb, kernel_name, **kwargs)
705.1s	1515	           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
705.1s	1516	  File "/usr/local/lib/python3.12/dist-packages/papermill/engines.py", line 374, in execute_notebook
705.1s	1517	    nb_man.notebook_complete()
705.1s	1518	  File "/usr/local/lib/python3.12/dist-packages/papermill/engines.py", line 77, in wrapper
705.1s	1519	    return func(self, *args, **kwargs)
705.1s	1520	           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
705.1s	1521	  File "/usr/local/lib/python3.12/dist-packages/papermill/engines.py", line 302, in notebook_complete
705.1s	1522	    self.save()
705.1s	1523	  File "/usr/local/lib/python3.12/dist-packages/papermill/engines.py", line 77, in wrapper
705.1s	1524	    return func(self, *args, **kwargs)
705.1s	1525	           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
705.1s	1526	  File "/usr/local/lib/python3.12/dist-packages/papermill/engines.py", line 153, in save
705.1s	1527	    write_ipynb(self.nb, self.output_path)
705.1s	1528	  File "/usr/local/lib/python3.12/dist-packages/papermill/iorw.py", line 486, in write_ipynb
705.1s	1529	    papermill_io.write(nbformat.writes(nb), path)
705.1s	1530	  File "/usr/local/lib/python3.12/dist-packages/papermill/iorw.py", line 102, in write
705.1s	1531	    return self.get_handler(path, extensions).write(buf, path)
705.1s	1532	           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
705.1s	1533	  File "/usr/local/lib/python3.12/dist-packages/papermill/iorw.py", line 215, in write
705.1s	1534	    f.write(buf)
705.1s	1535	OSError: [Errno 28] No space left on device
705.5s	1536	/bin/bash: line 19: echo: write error: No space left on device
708.0s	1537	/usr/local/lib/python3.12/dist-packages/mistune.py:435: SyntaxWarning: invalid escape sequence '\|'
708.0s	1538	  cells[i][c] = re.sub('\\\\\|', '|', cell)
708.3s	1539	/usr/local/lib/python3.12/dist-packages/nbconvert/filters/filter_links.py:36: SyntaxWarning: invalid escape sequence '\_'
708.3s	1540	  text = re.sub(r'_', '\_', text) # Escape underscores in display text
709.2s	1541	Traceback (most recent call last):
709.2s	1542	  File "/usr/local/lib/python3.12/dist-packages/traitlets/utils/importstring.py", line 35, in import_item
709.2s	1543	    pak = getattr(module, obj)
709.2s	1544	          ^^^^^^^^^^^^^^^^^^^^
709.2s	1545	AttributeError: module 'remove_papermill_header' has no attribute 'RemovePapermillHeader'
709.2s	1546	
709.2s	1547	The above exception was the direct cause of the following exception:
709.2s	1548	
709.2s	1549	Traceback (most recent call last):
709.2s	1550	  File "/usr/local/bin/jupyter-nbconvert", line 10, in <module>
709.2s	1551	    sys.exit(main())
709.2s	1552	             ^^^^^^
709.2s	1553	  File "/usr/local/lib/python3.12/dist-packages/jupyter_core/application.py", line 284, in launch_instance
709.2s	1554	    super().launch_instance(argv=argv, **kwargs)
709.2s	1555	  File "/usr/local/lib/python3.12/dist-packages/traitlets/config/application.py", line 1075, in launch_instance
709.2s	1556	    app.start()
709.2s	1557	  File "/usr/local/lib/python3.12/dist-packages/nbconvert/nbconvertapp.py", line 369, in start
709.2s	1558	    self.convert_notebooks()
709.2s	1559	  File "/usr/local/lib/python3.12/dist-packages/nbconvert/nbconvertapp.py", line 536, in convert_notebooks
709.2s	1560	    self.exporter = cls(config=self.config)
709.2s	1561	                    ^^^^^^^^^^^^^^^^^^^^^^^
709.2s	1562	  File "/usr/local/lib/python3.12/dist-packages/nbconvert/exporters/exporter.py", line 118, in __init__
709.2s	1563	    self._init_preprocessors()
709.2s	1564	  File "/usr/local/lib/python3.12/dist-packages/nbconvert/exporters/exporter.py", line 275, in _init_preprocessors
709.2s	1565	    self.register_preprocessor(preprocessor, enabled=True)
709.2s	1566	  File "/usr/local/lib/python3.12/dist-packages/nbconvert/exporters/exporter.py", line 236, in register_preprocessor
709.2s	1567	    preprocessor_cls = import_item(preprocessor)
709.2s	1568	                       ^^^^^^^^^^^^^^^^^^^^^^^^^
709.2s	1569	  File "/usr/local/lib/python3.12/dist-packages/traitlets/utils/importstring.py", line 37, in import_item
709.2s	1570	    raise ImportError("No module named %s" % obj) from e
709.2s	1571	ImportError: No module named RemovePapermillHeader
711.5s	1572	[NbConvertApp] Converting notebook __notebook__.ipynb to html
711.5s	1573	Traceback (most recent call last):
711.5s	1574	  File "/usr/local/lib/python3.12/dist-packages/nbformat/reader.py", line 19, in parse_json
711.5s	1575	    nb_dict = json.loads(s, **kwargs)
711.5s	1576	              ^^^^^^^^^^^^^^^^^^^^^^^
711.5s	1577	  File "/usr/lib/python3.12/json/__init__.py", line 346, in loads
711.5s	1578	    return _default_decoder.decode(s)
711.5s	1579	           ^^^^^^^^^^^^^^^^^^^^^^^^^^
711.5s	1580	  File "/usr/lib/python3.12/json/decoder.py", line 338, in decode
711.5s	1581	    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
711.5s	1582	               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
711.5s	1583	  File "/usr/lib/python3.12/json/decoder.py", line 354, in raw_decode
711.5s	1584	    obj, end = self.scan_once(s, idx)
711.5s	1585	               ^^^^^^^^^^^^^^^^^^^^^^
711.5s	1586	json.decoder.JSONDecodeError: Unterminated string starting at: line 3789 column 7 (char 182108)
711.5s	1587	
711.5s	1588	The above exception was the direct cause of the following exception:
711.5s	1589	
711.5s	1590	Traceback (most recent call last):
711.5s	1591	  File "/usr/local/bin/jupyter-nbconvert", line 10, in <module>
711.5s	1592	    sys.exit(main())
711.5s	1593	             ^^^^^^
711.5s	1594	  File "/usr/local/lib/python3.12/dist-packages/jupyter_core/application.py", line 284, in launch_instance
711.5s	1595	    super().launch_instance(argv=argv, **kwargs)
711.5s	1596	  File "/usr/local/lib/python3.12/dist-packages/traitlets/config/application.py", line 1075, in launch_instance
711.5s	1597	    app.start()
711.5s	1598	  File "/usr/local/lib/python3.12/dist-packages/nbconvert/nbconvertapp.py", line 369, in start
711.5s	1599	    self.convert_notebooks()
711.5s	1600	  File "/usr/local/lib/python3.12/dist-packages/nbconvert/nbconvertapp.py", line 541, in convert_notebooks
711.5s	1601	    self.convert_single_notebook(notebook_filename)
711.5s	1602	  File "/usr/local/lib/python3.12/dist-packages/nbconvert/nbconvertapp.py", line 506, in convert_single_notebook
711.5s	1603	    output, resources = self.export_single_notebook(notebook_filename, resources, input_buffer=input_buffer)
711.5s	1604	                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
711.5s	1605	  File "/usr/local/lib/python3.12/dist-packages/nbconvert/nbconvertapp.py", line 435, in export_single_notebook
711.5s	1606	    output, resources = self.exporter.from_filename(notebook_filename, resources=resources)
711.5s	1607	                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
711.5s	1608	  File "/usr/local/lib/python3.12/dist-packages/nbconvert/exporters/exporter.py", line 190, in from_filename
711.5s	1609	    return self.from_file(f, resources=resources, **kw)
711.5s	1610	           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
711.5s	1611	  File "/usr/local/lib/python3.12/dist-packages/nbconvert/exporters/exporter.py", line 208, in from_file
711.5s	1612	    return self.from_notebook_node(nbformat.read(file_stream, as_version=4), resources=resources, **kw)
711.5s	1613	                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
711.5s	1614	  File "/usr/local/lib/python3.12/dist-packages/nbformat/__init__.py", line 174, in read
711.5s	1615	    return reads(buf, as_version, capture_validation_error, **kwargs)
711.5s	1616	           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
711.5s	1617	  File "/usr/local/lib/python3.12/dist-packages/nbformat/__init__.py", line 92, in reads
711.5s	1618	    nb = reader.reads(s, **kwargs)
711.5s	1619	         ^^^^^^^^^^^^^^^^^^^^^^^^^
711.5s	1620	  File "/usr/local/lib/python3.12/dist-packages/nbformat/reader.py", line 75, in reads
711.5s	1621	    nb_dict = parse_json(s, **kwargs)
711.5s	1622	              ^^^^^^^^^^^^^^^^^^^^^^^
711.5s	1623	  File "/usr/local/lib/python3.12/dist-packages/nbformat/reader.py", line 25, in parse_json
711.5s	1624	    raise NotJSONError(message) from e
711.5s	1625	nbformat.reader.NotJSONError: Notebook does not appear to be JSON: '{\n "cells": [\n  {\n   "cell_type": "m...
import asyncio
import logging
import socket
import time

import numpy as np
import pyaudio


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def listen_to_F5TTS(text, server_ip="localhost", server_port=9998):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    await asyncio.get_event_loop().run_in_executor(None, client_socket.connect, (server_ip, int(server_port)))

    start_time = time.time()
    first_chunk_time = None

    async def play_audio_stream():
        nonlocal first_chunk_time
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32, channels=1, rate=24000, output=True, frames_per_buffer=2048)

        try:
            while True:
                data = await asyncio.get_event_loop().run_in_executor(None, client_socket.recv, 8192)
                if not data:
                    break
                if data == b"END":
                    logger.info("End of audio received.")
                    break

                audio_array = np.frombuffer(data, dtype=np.float32)
                stream.write(audio_array.tobytes())

                if first_chunk_time is None:
                    first_chunk_time = time.time()

        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

        logger.info(f"Total time taken: {time.time() - start_time:.4f} seconds")

    try:
        data_to_send = f"{text}".encode("utf-8")
        await asyncio.get_event_loop().run_in_executor(None, client_socket.sendall, data_to_send)
        await play_audio_stream()

    except Exception as e:
        logger.error(f"Error in listen_to_F5TTS: {e}")

    finally:
        client_socket.close()


if __name__ == "__main__":
    text_to_send = "As a Reader assistant, I'm familiar with new technology. which are key to its improved performance in terms of both training speed and inference efficiency. Let's break down the components"

    asyncio.run(listen_to_F5TTS(text_to_send))
