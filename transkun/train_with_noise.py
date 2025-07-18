import os
import sys
import random
import wandb
import numpy as np
import math
import torch
import argparse
import time
import scipy.signal
import copy
from torch.utils.tensorboard.writer import SummaryWriter
from . import Data
from .TrainUtil import *
import moduleconf
import torch.distributed as dist
import torch.multiprocessing as mp

def inject_noise(audio, noise_type, snr_db):
    """Injects white or pink noise into an audio signal."""
    if audio.size == 0:
        return audio
    signal_power = np.mean(audio**2)
    if signal_power == 0:
        return audio
    signal_db = 10 * np.log10(signal_power)
    noise_db = signal_db - snr_db
    noise_power = 10**(noise_db / 10)
    if noise_type == 'white':
        noise = np.random.normal(0, np.sqrt(noise_power), audio.shape)
    elif noise_type == 'pink':
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        zi = np.random.normal(0, 1, size=(max(len(a), len(b)) - 1,))
        pink_noise, _ = scipy.signal.lfilter(b, a, np.random.normal(0, 1, audio.shape[0]), zi=zi)
        noise_power_pink = np.mean(pink_noise**2)
        if noise_power_pink == 0:
            return audio
        noise = pink_noise * np.sqrt(noise_power / noise_power_pink)
    else:
        raise ValueError("Unsupported noise type")
    return audio + noise.astype(np.float32)

def train(workerId, nWorker, filename, runSeed, args):
    parallel = True
    if nWorker == 1:
        parallel = False
    if parallel:
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = args.master_port
        dist.init_process_group('nccl', rank=workerId, world_size=nWorker)
    
    device = torch.device("cuda:"+str(workerId%torch.cuda.device_count()) if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    random.seed(workerId+int(time.time()))
    np.random.seed(workerId + int(time.time()))
    torch.manual_seed(workerId+int(time.time()))
    torch.cuda.manual_seed(workerId+ int(time.time()))
    
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True 
        torch.backends.cudnn.allow_tf32 = True 
        
    confManager = moduleconf.parseFromFile(args.modelConf)
    TransKun = confManager["Model"].module.TransKun
    conf = confManager["Model"].config
    
    if workerId == 0:
        # if the saved file does not exist
        if not os.path.exists(filename):
            print("initializing the model...")
            startEpoch, startIter, model,  lossTracker, best_state_dict, optimizer, lrScheduler = initializeCheckpoint(
                                TransKun,
                                device = device,
                                max_lr= args.max_lr,
                                weight_decay = args.weight_decay,
                                nIter = args.nIter,
                                conf = conf)
            save_checkpoint(filename, startEpoch, startIter, model,  lossTracker, best_state_dict, optimizer, lrScheduler)

    if parallel:
        dist.barrier()
        
    startEpoch, startIter, model,  lossTracker, best_state_dict, optimizer, lrScheduler= load_checkpoint(TransKun, conf, filename,device)
    print("#{} loaded".format(workerId))

    if workerId == 0:
        print("loading dataset....")

    datasetPath = args.datasetPath
    datasetPicklePath = args.datasetMetaFile_train
    datasetPicklePath_val = args.datasetMetaFile_val

    dataset = Data.DatasetMaestro(datasetPath, datasetPicklePath)
    datasetVal = Data.DatasetMaestro(datasetPath, datasetPicklePath_val)

    print("#{} loaded".format(workerId))

    if workerId == 0:
        writer = SummaryWriter(filename+".log")
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    globalStep = startIter
    # create dataloader

    # this iterator should be constructed each time
    batchSize = args.batchSize

    if args.hopSize is None:
        hopSize = conf.segmentHopSizeInSecond
    else:
        hopSize = args.hopSize

    if args.chunkSize is None:
        chunkSize =  conf.segmentSizeInSecond
    else:
        chunkSize = args.chunkSize

    gradNormHist = MovingBuffer(initValue = 40, maxLen = 10000)

    augmentator = None
    if args.augment:
        augmentator = Data.AugmentatorAudiomentations(sampleRate=44100, noiseFolder = args.noiseFolder, convIRFolder = args.irFolder)

    for epoc in range(startEpoch, 1000000):
        dataIter = Data.DatasetMaestroIterator(dataset, hopSize, chunkSize, seed = epoc*100+runSeed, augmentator = augmentator, notesStrictlyContained=False)
        if parallel:
            sampler = torch.utils.data.distributed.DistributedSampler(dataIter)
            sampler.set_epoch(epoc)
            dataloader= torch.utils.data.DataLoader(dataIter, batch_size = batchSize, collate_fn = Data.collate_fn_batching, num_workers=args.dataLoaderWorkers, sampler = sampler, drop_last = True, prefetch_factor = max(4, args.dataLoaderWorkers))
        else:
            dataloader= torch.utils.data.DataLoader(dataIter, batch_size = batchSize, collate_fn = Data.collate_fn_batching, num_workers=args.dataLoaderWorkers, shuffle=True, prefetch_factor = max(4, args.dataLoaderWorkers))
        lossAll = []
        globalStepWarmupCutoff = globalStep+500
        for idx, batch in enumerate(dataloader):
            if workerId ==0:
                currentLR = [p["lr"] for p in optimizer.param_groups][0]
                writer.add_scalar(f'Optimizer/lr', currentLR , globalStep)
            computeStats = False 
            if idx % 40 == 0:
                computeStats =True
            t1 = time.time()
            model.train()
            optimizer.zero_grad()
            totalBatch= torch.zeros(1).cuda()
            totalLoss = torch.zeros(1).cuda()
            totalLen = torch.zeros(1).cuda()
            totalGT= torch.zeros(1).cuda()
            totalEst = torch.zeros(1).cuda()
            totalCorrect = torch.zeros(1).cuda()
            totalGTFramewise = torch.zeros(1).cuda()
            totalEstFramewise = torch.zeros(1).cuda()
            totalCorrectFramewise = torch.zeros(1).cuda()
            totalSEVelocity = torch.zeros(1).cuda()
            totalSEOF= torch.zeros(1).cuda()
            notesBatch = batch["notes"]
            audioSlices = batch["audioSlices"].to(device)
            # Inject noise into training audio
            if args.noise_type is not None:
                audioSlices_np = audioSlices.cpu().numpy()
                for i in range(audioSlices_np.shape[0]):
                    audioSlices_np[i] = inject_noise(audioSlices_np[i], args.noise_type, args.snr_db)
                # Check for NaN or Inf in audioSlices_np
                if np.isnan(audioSlices_np).any() or np.isinf(audioSlices_np).any():
                    print("NaN or Inf detected in audioSlices_np after noise injection! Exiting.")
                    exit(1)
                audioSlices = torch.from_numpy(audioSlices_np).to(device)
            audioLength = audioSlices.shape[1]/model.conf.fs

            logp = model.log_prob(audioSlices, notesBatch)
            loss = (-logp.sum(-1).mean())
            (loss/50).backward()
            totalBatch = totalBatch + 1
            totalLen = totalLen + audioLength
            totalLoss = totalLoss + loss.detach()
            if computeStats:
                with torch.no_grad():
                    model.eval()
                    stats = model.computeStats(audioSlices, notesBatch)
                    stats2 = model.computeStatsMIREVAL(audioSlices, notesBatch)
            else:
                # Set default values when stats are not computed
                stats = {"nGTFramewise": 0, "nEstFramewise": 0, "nCorrectFramewise": 0, "seVelocityForced": 0, "seOFForced": 0}
                stats2 = {"nGT": 0, "nEst": 0, "nCorrect": 0}

            totalGT = totalGT+ stats2["nGT"]
            totalEst= totalEst + stats2["nEst"]
            totalCorrect= totalCorrect+ stats2["nCorrect"]
            totalGTFramewise = totalGTFramewise+ stats["nGTFramewise"]
            totalEstFramewise = totalEstFramewise + stats["nEstFramewise"]
            totalCorrectFramewise = totalCorrectFramewise+ stats["nCorrectFramewise"]
            totalSEVelocity = totalSEVelocity + stats["seVelocityForced"]
            totalSEOF = totalSEOF + stats["seOFForced"]

            # checkNoneGradient(model)
            if parallel:
                dist.all_reduce(totalLoss.data)
                dist.all_reduce(totalLen.data)
                dist.all_reduce(totalBatch.data)
                if computeStats:
                    dist.all_reduce(totalGT.data)
                    dist.all_reduce(totalEst.data)
                    dist.all_reduce(totalCorrect.data)
                    dist.all_reduce(totalGTFramewise.data)
                    dist.all_reduce(totalEstFramewise.data)
                    dist.all_reduce(totalCorrectFramewise.data)
                    dist.all_reduce(totalSEOF.data)
                    dist.all_reduce(totalSEVelocity.data)
                average_gradients(model, totalLen, parallel)
            loss = totalLoss/totalLen
            curClipValue = gradNormHist.getQuantile(args.gradClippingQuantile)
            totalNorm = torch.nn.utils.clip_grad_norm_(model.parameters(), curClipValue)
            gradNormHist.step(totalNorm.item())
            optimizer.step()
            try:
                if globalStep> globalStepWarmupCutoff:
                    lrScheduler.step()
            except:
                # continue after the final iteration
                pass
            if workerId == 0:
                t2 = time.time()
                print("epoch:{} progress:{:0.3f} step:{}  loss:{:0.4f} gradNorm:{:0.2f} clipValue:{:0.2f} time:{:0.2f} ".format(epoc , idx/len(dataloader), 0,loss.item(),totalNorm.item(), curClipValue, t2-t1))
                writer.add_scalar(f'Loss/train', loss.item(), globalStep)
                writer.add_scalar(f'Optimizer/gradNorm', totalNorm.item(), globalStep)
                writer.add_scalar(f'Optimizer/clipValue', curClipValue, globalStep)
                wandb.log({'loss': loss.item(), 'gradNorm': totalNorm.item(), 'clipValue': curClipValue, 'epoch': epoc, 'step': globalStep})
                if computeStats:
                    nGT = totalGT.item()+1e-4
                    nEst = totalEst.item()+1e-4
                    nCorrect = totalCorrect.item()+1e-4
                    precision = nCorrect/nEst
                    recall = nCorrect/nGT
                    f1 = 2* precision*recall/(precision+recall)              
                    print("nGT:{} nEst:{} nCorrect:{}".format(nGT, nEst, nCorrect))
                    writer.add_scalar(f'Loss/train_f1', f1, globalStep)
                    writer.add_scalar(f'Loss/train_precision', precision, globalStep)
                    writer.add_scalar(f'Loss/train_recall', recall, globalStep)
                    wandb.log({'f1': f1, 'precision': precision, 'recall': recall, 'epoch': epoc, 'step': globalStep})
                    nGTFramewise = totalGTFramewise.item()+1e-4
                    nEstFramewise = totalEstFramewise.item()+1e-4
                    nCorrectFramewise = totalCorrectFramewise.item()+1e-4
                    precisionFrame = nCorrectFramewise/nEstFramewise
                    recallFrame = nCorrectFramewise/nGTFramewise
                    f1Frame = 2* precisionFrame*recallFrame/(precisionFrame+recallFrame)              
                    mseVelocity = totalSEVelocity.item()/nGT
                    mseOF = totalSEOF.item()/nGT
                    writer.add_scalar(f'Loss/train_f1_frame', f1Frame, globalStep)
                    writer.add_scalar(f'Loss/train_precision_frame', precisionFrame, globalStep)
                    writer.add_scalar(f'Loss/train_recall_frame', recallFrame, globalStep)
                    writer.add_scalar(f'Loss/train_mse_velocity', mseVelocity, globalStep)
                    writer.add_scalar(f'Loss/train_mse_OF', mseOF, globalStep)
                    wandb.log({'f1_frame': f1Frame, 'precision_frame': precisionFrame, 'recall_frame': recallFrame, 'mse_velocity': mseVelocity, 'mse_of': mseOF, 'epoch': epoc, 'step': globalStep})
                    print("f1:{} precision:{} recall:{}".format(f1, precision, recall))
                    print("f1Frame:{} precisionFrame:{} recallFrame:{}".format(f1Frame, precisionFrame, recallFrame))
                    print("mseVelocity:{} mseOF:{}".format(mseVelocity, mseOF))
                if math.isnan(loss.item()):
                    exit()
                lossAll.append(loss.item())
                if idx%2000 ==1999:
                    save_checkpoint(filename, epoc+1, globalStep+1, model,  lossTracker, best_state_dict, optimizer, lrScheduler)
                    print("saved")
            globalStep+= 1
        if workerId == 0:
            print("Validating...")
        torch.cuda.empty_cache()
        dataIterVal = Data.DatasetMaestroIterator(datasetVal, hopSizeInSecond = conf.segmentHopSizeInSecond, chunkSizeInSecond = chunkSize, notesStrictlyContained=False, seed = runSeed+epoc*100) 
        if parallel:
            samplerVal = torch.utils.data.distributed.DistributedSampler(dataIterVal)
            dataloaderVal= torch.utils.data.DataLoader(dataIterVal, batch_size = 2*batchSize, collate_fn = Data.collate_fn , num_workers=args.dataLoaderWorkers, sampler = samplerVal)
        else:
            dataloaderVal= torch.utils.data.DataLoader(dataIterVal, batch_size = 2*batchSize, collate_fn = Data.collate_fn , num_workers=args.dataLoaderWorkers, shuffle=True)
        model.eval()
        valResult = doValidation(model, dataloaderVal, parallel = parallel, device = device)
        nll = valResult["meanNLL"]
        f1 = valResult["f1"]
        torch.cuda.empty_cache()
        if workerId == 0:
            lossAveraged = sum(lossAll)/len(lossAll)
            lossAll = []
            lossTracker['train'].append(lossAveraged)
            lossTracker['val'].append(f1)
            print('result:', valResult)
            for key in valResult:
                writer.add_scalar('val/'+ key, valResult[key], epoc)
                wandb.log({f'val/{key}': valResult[key], 'epoch': epoc})
            if f1 >= max(lossTracker['val'])*1.00:
                print('best updated')
                best_state_dict = copy.deepcopy(model.state_dict())
            save_checkpoint(filename, epoc+1, globalStep+1, model,  lossTracker, best_state_dict, optimizer, lrScheduler)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Perform Training with Noise and wandb logging")
    parser.add_argument('saved_filename')
    parser.add_argument('--nProcess', help="# of processes for parallel training", required = True, type=int)
    parser.add_argument('--master_addr', help="master address for distributed training ", default = '127.0.0.1')
    parser.add_argument('--master_port', help='master port number for distributed training', default = "29500")
    parser.add_argument('--allow_tf32', action = "store_true")
    parser.add_argument('--datasetPath', required = True)
    parser.add_argument('--datasetMetaFile_train', required = True)
    parser.add_argument('--datasetMetaFile_val', required = True)
    parser.add_argument('--batchSize', default=4, type=int)
    parser.add_argument('--hopSize', required=False, type=float)
    parser.add_argument('--chunkSize', required=False, type=float)
    parser.add_argument('--dataLoaderWorkers', default = 2, type=int)
    parser.add_argument('--gradClippingQuantile', default = 0.8, type=float)
    parser.add_argument('--max_lr', default = 2e-4, type=float)
    parser.add_argument('--weight_decay', default = 1e-4, type=float)
    parser.add_argument('--nIter', default= 180000,type = int)
    parser.add_argument('--modelConf', required=True, help = "the path to the model conf file")
    parser.add_argument('--augment',  action ="store_true", help="do data augmentation")
    parser.add_argument('--noiseFolder',  required = False)
    parser.add_argument('--irFolder',  required = False)
    parser.add_argument('--noise_type', default=None, choices=['white', 'pink'], help='Type of noise to inject (white or pink)')
    parser.add_argument('--snr_db', default=None, type=float, help='SNR (dB) for noise injection')
    parser.add_argument('--wandb_project', default='transkun-noise', help='wandb project name')
    parser.add_argument('--wandb_run_name', default=None, help='wandb run name')
    args = parser.parse_args()
    num_processes = args.nProcess
    saved_filename = args.saved_filename
    runSeed = int(time.time())
    if num_processes == 1:
        train(0, 1, saved_filename, runSeed, args)
    else:
        mp.spawn(fn=train, args=(num_processes, saved_filename, runSeed, args),  nprocs = num_processes, join=True, daemon=False) 