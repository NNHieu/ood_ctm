# python eval_ood.py -m run_name=custom_aug/cifar10_wrn in_dataset=cifar10 net=wrn detector='glob(*,exclude=base)' test_bs=128 save2csv=T
# # 
# python eval_ood.py -m run_name=custom_aug/cifar10_densenet in_dataset=cifar10 net=densenet detector='glob(*,exclude=base)' test_bs=128 save2csv=T
# # 
# python eval_ood.py -m run_name=custom_aug/cifar100_wrn in_dataset=cifar100 net=wrn detector='glob(*,exclude=base)' test_bs=128 save2csv=T
# # 
# python eval_ood.py -m run_name=custom_aug/cifar100_densenet in_dataset=cifar100 net=densenet detector='glob(*,exclude=[base])' test_bs=128 save2csv=T
# #

# python eval_ood.py in_dataset=cifar10 net=vit run_name=cifar10_vit detector=msp test_bs=400 save2csv=True num_to_avg=10
# python eval_ood.py in_dataset=cifar10 net=vit run_name=cifar10_vit detector=msp test_bs=400 save2csv=True num_to_avg=10
# python eval_ood.py in_dataset=cifar100 net=vit run_name=cifar100_vit detector=msp test_bs=400 save2csv=True num_to_avg=10
# python eval_ood.py in_dataset=cifar100 net=vit run_name=cifar100_vit detector=ctm test_bs=300 save2csv=True num_to_avg=10 detector.cache_class_means="cached/vit_cifar100"

# python eval_ood.py in_dataset=cifar10 net=vit run_name=cifar10_vit detector=knn test_bs=400 save2csv=True num_to_avg=10 detector.cache="cached/vit_cifar10_knn.npy"
# python eval_ood.py in_dataset=cifar100 net=vit run_name=cifar100_vit detector=knn test_bs=300 save2csv=True num_to_avg=10 detector.cache="cached/vit_cifar100_knn.npy"

# python eval_ood.py in_dataset=cifar10 net=vit run_name=cifar10_vit detector=knn test_bs=400 save2csv=True num_to_avg=10 detector.cache="cached/vit_cifar10_knn.npy"
# python eval_ood.py in_dataset=cifar100 net=vit run_name=cifar100_vit detector=knn test_bs=300 save2csv=True num_to_avg=10 detector.cache="cached/vit_cifar100_knn.npy"

python eval_ood.py in_dataset=cifar10 net=vit run_name=cifar10_vit detector=mahalanobis test_bs=100 save2csv=True num_to_avg=5 detector.cache="cached/vit_cifar10_mahalanobis.pt"
python eval_ood.py in_dataset=cifar100 net=vit run_name=cifar100_vit detector=mahalanobis test_bs=100 save2csv=True num_to_avg=5 detector.cache="cached/vit_cifar100_mahalanobis.pt"


# python eval_ood.py in_dataset=cifar10 net=vit run_name=cifar10_vit detector=knn test_bs=400 save2csv=True num_to_avg=10


# python eval_ood.py in_dataset=cifar10 net=vit run_name=cifar10_vit detector=osa test_bs=200 save2csv=True num_to_avg=5
# python eval_ood.py in_dataset=cifar100 net=vit run_name=cifar100_vit detector=osa test_bs=200 save2csv=True num_to_avg=5
