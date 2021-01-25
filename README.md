﻿
Generative Adversarial Network on Style Imitation

Remy´ Sabathier Omar Soua¨ıdi

CentraleSupelec´ CentraleSupelec´ Paris-Saclay University ENS Paris-Saclay (MVA)

Abstract ![](Souaidi\_Sabathier\_DL\_Project%20(1).001.png)

Generative Adversarial Networks (called GANs) is a type of deep learning models introduced by Ian Goodfel- low and other researchers in 2014. In this type of models, two neural networks are competing against each other’s in order to learn how to generate samples with the same statistics as the original distribution. On its early release, GAN were identified as highly unstable during the training (because of the competition between the two networks) and hard to evaluate. These two issues are still active domain in Artificial Intelligence: we propose to study the key aspects

of these two problems by participating to an open competi- Figure 1. Example from the Monnet painting training set.

(256x256x3) in TF records format.

tion hosted on Kaggle: ‘I’m something of a painter myself’.

Dataset Computing Units To train our GAN, the com-

\1. Introduction petition proposes a collection of 300 Monet paintings. We then have 7363 images that can be used to test the style

The competition is a style transfer challenge based on

transfer. One interesting auxiliary aspect of the competition Monnet paintings. As such, given a dataset of real-life im-

is the use of the TF records format for the dataset and TPU ages, we are asked to design a GAN that can give the style

processors for training.

of Monnet paintings to an image of our choice. After imple-

menting a model, we can send our proposition to measure 2. Evaluation

the performance of our network compared to other partic-

ipants (as we are writing this description, 221 teams are Evaluate numerically the results of a generative model is enrolled). A constraint on the training time of the model not as straight-forward as classical neural networks perfor- is imposed (less than 3 hours training time on TPUs or 5 mance evaluation where we have metrics (for classification hours on GPUs). or regression task for instance). In generative models, the

objective of the generator is to learn how to generate, from Evaluation In generative adversarial learning, evaluating a random latent vector z, a sample belonging to a target dis- a model is not as simple as in classification for instance tribution. However, in the majority of cases, the target dis- where the progression of a model can be easily quanti- tributionisnotwelldefined. Inthecompetitioncontext, it is fied witch metrics. Evaluating GANs performance is still Monet painting images. We analyzed two different metrics considered as active research subject. This project is also to understand how evaluation is performed in this context. motivated on understanding the current most used metrics 2.1. Inception Score

for GAN evaluation (Frechet´ Inception Distance [2], In-

ception Score [3]). The Kaggle competition propose to The Inception Score was introduced in [\[5\]](#_page7_x50.11_y346.54) as a firstmet- evaluate candidate on the MiFID: memorization informed ric that correlates with human judgement for GAN evalu- Frechet´ Inception Distance, which is a modified version ation. They start by applying the Inception model on the of the Frechet´ Inception Distance which takes training set generated images to get a label distribution y for each sam- memorization into account. ple x. The loss is then based on the two principles:

1. The conditional probability p(yjx) should have a high where mr is the memorization penalty, and s is definedas 1 entropy since we want to generate realistic image that minus the average cosine similarity between a fake sample triggers only one label. sf and the full training set. As such, models that memo- P rizethetrainingsetwillhaveastrongmemorizationpenalty.
1. The sum of all conditional probability i p(yijxi) TheMiFIDisthemetricusedtoevaluatemodelsintheKag- (called marginal probability) should have a low en- gle competition we attended.

tropy since we want to generate diverse samples.

The inception score then computes the KL-Divergence 3. Models

between the marginal probability and each conditional We decided to focus only on generative model, which probability and takes the exponential of the average. Ac- are specifically designed for the competition. Generative cording to the 2 mentioned principles, if the GAN generates models are generally difficult to train due to the inherent diverse and realistic samples, we should expect a very high instability of adversarial training. The Kaggle competition score. we attended is based on Monet painting generation from

A firstlimit of the inception score is that it is only appli- real life images, which is an image-to-image translation cable on the training set used for the training of the incep- task on an unpaired dataset.

tion network, which is ImageNet. As well, the score doesn’t

use the training images at all for evaluating how well the 3.1. Cycle-GAN

GAN is able to generate fake samples.

CycleGAN[\[7\]](#_page7_x50.11_y414.29)isastandardgenerativemodelforimage-

2. Frechet Inception Distance (FID) to-image problems on unpaired dataset. In the original pa- per, Monet painting generation from standard image is pre-

In [\[3](#_page7_x50.11_y267.84)], the authors tries to address the inception scores is-

sented as a application of their method.

sues by proposing the Frechet Inception Distance (FID).

Like the inception score, the method is based on the pre-

trained Inception model, but they use the output of an in- Objective Let’s consider P and M, the domain space as- termediate layer with 2048 features. For each training sam- sociated to the real photos and the Monet paintings. The ples and a large batch of fake samples, they compute the ac- model includes two Generators GP and GM that maps a tivation features and approximate the distribution for fake input to the domain P or M respectively. As an adversarial and real by a multivariate gaussian N (real ;real ) and model, we also includes two discriminators DP and DM . N (fake;fake). The score is then the Frechet Distance The objective of DM is to distinguish between real Monet between the two distributions: painting and fake samples generated from GM , and the ob-

jective of DP is to distinguish between real pictures and p fakes from GP .![](Souaidi\_Sabathier\_DL\_Project%20(1).002.png)

jjreal   fakejj2 + Tr(real + fake   2 real fake)

The FID score better correlates with the quality and di- Loss is divided in 3 parts:

versity of fake images and it can also be used on GAN that 1. P Adversarial Loss. G tries to generate realistic are not trained on ImageNet dataset (as long as the features fake Monet painting fromMpicture input, whereas D

detectedbyInceptionintermediatelayersaremeaningfulon tries to make the difference between real Monet m andM this training set). fake ones G (p):

M

3. Memorized FID (Mi-FID)

The main issue of the previous methods for evaluating [1  log(D

GAN performance is the fact models can easily achieve Ladv1 = EpP M (GM (p)))]+ strongperformancebymemorizingasmallsubsetofthetar- EmM [log(DM (m))]

get distribution. To penalize the performance of these mod-

els, [\[2\]](#_page7_x50.11_y232.15) introduced the Memorization-Informed Frechet 2. M Adversarial Loss. Same, with real picture genera- Inception Distance (MiFID). The score between the fake tion:

distribution Df and the real distribution Dr is computed by

Ladv2 = EmM [1  log(DP (GP (m)))]+

EpP [log(DP (p))] MiFID(Df ;Dr) = mr(Df ;Dr):FID(Df ;Dr)

1

mr(Df ;Dr) = s(Df ;Dr) +  3. Cycle consistency loss. Mainly for regularization pur-![](Souaidi\_Sabathier\_DL\_Project%20(1).003.png)

poses, theyforcethegeneratortobeabletoreconstruct

![](Souaidi\_Sabathier\_DL\_Project%20(1).004.png)![](Souaidi\_Sabathier\_DL\_Project%20(1).005.png)

Figure 2. Cycle-Gan Architecture selected. top: Generator. bottom: Discriminator

an image, on the same principle that for autoencoder of sampled latent variables to construct variant-concatenate training: inputs, transferring network sampling to input domain sam- pling. By exploring full posteriors, Bayesian CycleGAN

enhances generator training to resist crazy learning discrim- Lcycle = EmM [jjGM (GP (m))   mjj]+ inator, and therefore alleviates the risk of mode collapse,

E [jjG (G (p))   pjj] boosting realistic color generating and multimodal distribu- pP P M tion learning. The proposed Bayesian CycleGAN models

thetruedatadistributionmoreaccuratelybyfullyrepresent-

The final loss is the sum of these three elements, with ing the posterior distribution over the parameters of both generally a coefficientapplied on the cycle term. generator and discriminator. More details about the model

can be found in the original paper [\[6](#_page7_x50.11_y380.42)].

Architecture In the original paper, the authors used for

the generators a convolutional network with residual layers 3.3. U-GAT-IT

in the bottleneck (Fig.[5](#_page6_x84.37_y710.80)). We add skip connections to the U-GAT-IT (Unsupervised Generative Attentional Net- original implementation to achieve a better spatial recon- works With Adaptive Layer-Instance Normalization for struction of the image and stabilize the gradient propaga- Image-to-Image Translation) [\[4\]](#_page7_x50.11_y311.99) is a novel method for un- tion during training. For the decoder, we keep the original supervised image-to-image translation, which incorporates architecture as well which is a simple sequential convolu- a new attention module and a new learnable normalization tional network with 5 stacked convolutions and a 2D aver- function in an end-to-end manner.

age pooling at the end. Layer normalization [\[1\]](#_page7_x50.11_y209.28) is added This model guides the translation [Source Image (Nor- after convolutional layers (except input and output). mal Picture) to Target Image (Monet Painting)] to focus on

3.2. Bayesian Cycle-Gan more important regions and ignore minor regions by distin- guishing between source and target domains based on the

To solve the stability issue in Cycle-GAN, Bayesian attention map obtained by the auxiliary classifier. These Cycle-GAN [\[6\]](#_page7_x50.11_y380.42) were proposed. They introduce a latent attention maps (figure 3) are embedded into the generator space for exploring the full posteriors. Specifically, they and discriminator to focus on semantically important areas, combine source images with corresponding certain volume thus facilitating the shape transformation. While the atten-

![](Souaidi\_Sabathier\_DL\_Project%20(1).006.png)![](Souaidi\_Sabathier\_DL\_Project%20(1).007.png)

Figure 3. Visualization of the attention maps for 4 differents normal pictures. The firstone (second row) represent the attention map related to the identity A2A. The fourth one to the translation A2B and the sixth one to the cycle A2B2A

tion map in the generator induces the focus on areas that flexibly control the amount of change in shape and texture. specificallydistinguish between the two domains, the atten- As a result, the model, without modifying the model archi- tion map in the discriminator helps fine-tuning by focusing tecture or the hyper-parameters, can perform image trans- on the difference between real image and fake image in tar- lation tasks not only requiring holistic changes but also re- get domain. quiring large shape changes.

In addition to the attentional mechanism, they find that

the choice of the normalization function has a significant 3.3.1 Architecture

impact on the quality of the transformed results. They pro-

pose an Adaptive LayerInstance Normalization (AdaLIN), Descriptionismainlytakenfrom theoriginalpaper [\[4](#_page7_x50.11_y311.99)]. The whose parameters are learned from datasets during train- architecture of the model can be seen in figure4.

ing time by adaptively selecting a proper ratio between In-

stance normalization (IN) and Layer Normalization (LN). Generator Let x 2 fXs;Xtg represent a sample from The AdaLIN function helps their attention-guided model to the source and the target domain. The translation model

![](Souaidi\_Sabathier\_DL\_Project%20(1).008.png)![](Souaidi\_Sabathier\_DL\_Project%20(1).009.png)

Figure 4. The model architecture of U-GAT-IT. Taken from [\[4](#_page7_x50.11_y311.99)].

Gs!t consists of an Encoder Es and a decoder Gt and pooling and global max pooling. By exploiting ws, we

k

an auxiliary classifier  where (x) represent the prob- can calculate a set of domain specific attention feature ability that x comes froms X . sLet Esk(x) be the k-th as(x) = ws Es(x) = fws Es =1 k ng where

k k

s

activation map of the encoder. The auxiliary classifier n is the number of encoded features maps. Then, the is trained to learn the weight of the k-th feature map translation model Gs!t becomes equal to Gt(as(x)). The for the source domain, ws by using the global average residual blocks with AdaLIN whose parameters and

k

are dynamically computed by a fully connected layer from Hyperparameters The CycleGAN model is trained with the attention map. a batch size of 1 and 100 epochs.

The Bayesian CycleGAN model is trained with a batch size AdaLIN (a; ; ) = (a^ ^(a )) + ; of 1 and 40 epochs. For every 10 epochs completed, we

I + (1   )  L generated the target images for the kaggle competition and

(1) we took the best score.

The U-GAT-IT model is trained with a batch size of 1 and a^ = pa   I ;a^L = pa   L (2) 60 epochs. For every 10 epochs completed, we generated![](Souaidi\_Sabathier\_DL\_Project%20(1).010.png)![](Souaidi\_Sabathier\_DL\_Project%20(1).011.png)

I I2 +  L2 +  the target images for the kaggle competition and we took

the best score.

where a^I , a^L andclipI[0;and1] (L arechannel-wise,) layer- [com/ranery/Bayesian-](https://github.com/ranery/Bayesian-CycleGAN)was based on the original code[CycleGAN](https://github.com/ranery/Bayesian-CycleGAN) [(https://github.](https://github.com/ranery/Bayesian-CycleGAN)and [https: ](https://github.com/znxlwm/UGATIT-pytorch)wise mean and standard deviation respectively, and are [//github.com/znxlwm/UGATIT-pytorch ](https://github.com/znxlwm/UGATIT-pytorch)respec-

(3) TheimplementationofBayesianCycleGANandU-GAT-IT

parameters generated by the fully connected layer, is the tively).

learning rate and indicates the parameter update vector Unfortunately, we were unable to make the implementation (the gradient) determined by the optimizer. The values of TPU-compatible, then we decided to run the training on are constrained to the range of [0, 1] simply by imposing Colab Pro with a TESLA V100. It took 40 hours to train bounds at the parameter update step. the Bayesian CycleGAN for 40 epochs and 65 hours to

train the U-GAT-IT model for 60 epochs.

Discriminator Let x 2 fXt;Gs!t(Xs)g represent a

sample from the target domain and the translated source

domain. Similartoothertranslationmodels, thediscrimina- Competition results We sent 42 submission on the Kag- tor Dt which is a multi-scale model consists of an encoder gle competition. We performed our best result with a EDt , a classifier CD , and an auxiliary classifier D . Un- UGATIT model scored 17th out of 221 teams (at the mo- liketheothertranslationt models,bothDt(x) and (tx) are ment we write the report). The best score on the leader-

D

trained to predict whether x comes from Xt or Gs!t t(X ). board is 37.72 . Our fine-tuned cycleGAN architecture

s

Given a sample x, Dt(x) exploits the attention feature maps achieved as well reasonably good performance compared mapsDt ED (x) tthat is trained by Dt(x). Then, our discrim- to the other competitors.

a (x) = wD EDt (x) using w on the encoded feature

inator Dt(t x) becomes equal to CDt (aDt (x)).

|CycleGan Bayesian CycleGAN UGATIT|
| - |
|Mi-FID 48.39 55.08 39.32|
Dt

Table 1. Best score obtained on the competition for the different theNoteoriginal: Morepaperinformation[\[4](#_page7_x50.11_y311.99)]. about the model can be found in models (lower is better).

\4. Results

Visual Interpretation The visual results presented in Our study of generative models was oriented towards the Fig.[5 ](#_page6_x84.37_y710.80)illustrates that our models were able to capture the

Kaggle competition for Monet painting generation. The re- general style that Monet paintings have in common and re- sultswe presentinthis section weredirectlycomputedfrom produce it on a real-life picture. Through the study of the the competition host in order to rank propositions. There visual results, we noticed that input pictures from landscape was a computational time constraint of maximum 5 hours are generally better converted compared to city or people of training on a single GPU (3 hours if the model is trained sincethedistributionofMonetpaintingismainlycomposed on TPU). of this category.

Optimizer The original CycleGan architecture is trained 5. Conclusion

with Adam optimizer (lr = 0:0002). We tested a recent

new optimizer AdaBelief [\[8\]](#_page7_x50.11_y447.49) on our CycleGan architec- In this project, we investigated classical ([[7](#_page7_x50.11_y414.29)]) and state ture. Themethodissupposedtostabilizethetraining. How- of the art ([[4](#_page7_x50.11_y311.99)],[[6](#_page7_x50.11_y380.42)]) models for style transfer on a Monet ever, we were unable to get any improvement with this op- Painting dataset. The study was motivated through the timizer (best score : 80.48). participation of an open competition hosted on Kaggle. We The Bayesian CycleGan was also trained with Adamopti- analyzed the current methods for generative model evalua- mizer (lr = 0:0002). tion (Inception Score, FID, Mi-FID) and the limits to their U-GAT-IT was trained with Adamoptimizer (lr = 0:0001). interpretation. We noticed how unstable generative models

![](Souaidi\_Sabathier\_DL\_Project%20(1).012.png)![](Souaidi\_Sabathier\_DL\_Project%20(1).013.png)

Figure 5. Comparison of models with different input pictures. As the Mi-FID score suggested, we consider that UGATIT achieves the best results

are because of the adversarial loss used for the training. Our implementation of CycleGAN and recent UGATIT model achieved good performance on the leaderboard, which is confirmedby our visual analysis of their outputs.

Code The implementation of our custom Cycle-GAN and the other models used is available on the[ Github repository ](https://github.com/omarsou/GAN-Art-Monet)of our project.

References

1. J. L. Ba, J. R. Kiros, and G. E. Hinton. Layer normalization, 2016.[ 3](#_page2_x49.11_y71.00)
1. C.-Y. Bai, H.-T. Lin, C. Raffel, and W. Kan. A large-scale study on training sample memorization in generative model- ing, 2021.[ 2](#_page1_x49.11_y71.00)
1. M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, G. Klam- bauer, and S. Hochreiter. Gans trained by a two time- scale update rule converge to a nash equilibrium. CoRR, abs/1706.08500, 2017.[ 2](#_page1_x49.11_y71.00)
1. J. Kim, M. Kim, H. Kang, and K. Lee. U-gat-it: Unsupervised generative attentional networks with adaptive layer-instance normalization for image-to-image translation, 2020.[ 3,](#_page2_x49.11_y71.00)[ 4,](#_page3_x49.11_y71.00)[ 5,](#_page4_x49.11_y71.00)[ 6](#_page5_x49.11_y71.00)
1. T. Salimans, I. J. Goodfellow, W. Zaremba, V. Cheung,
   1. Radford, and X. Chen. Improved techniques for training gans. CoRR, abs/1606.03498, 2016.[ 1](#_page0_x49.11_y71.00)
1. H. You, Y. Cheng, T. Cheng, C. Li, and P. Zhou. Bayesian cycle-consistentgenerativeadversarialnetworksviamarginal- izing latent sampling, 2020.[ 3,](#_page2_x49.11_y71.00)[ 6](#_page5_x49.11_y71.00)
1. J. Zhu, T. Park, P. Isola, and A. A. Efros. Unpaired image-to- image translation using cycle-consistent adversarial networks. CoRR, abs/1703.10593, 2017.[ 2,](#_page1_x49.11_y71.00)[ 6](#_page5_x49.11_y71.00)
1. J. Zhuang, T. Tang, Y. Ding, S. Tatikonda, N. Dvornek, X. Pa- pademetris, and J. S. Duncan. Adabelief optimizer: Adapting stepsizes by the belief in observed gradients, 2020.[ 6](#_page5_x49.11_y71.00)
PAGE11
