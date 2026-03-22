const { expect } = require("chai");
const { ethers } = require("hardhat");
const { time } = require("@nomicfoundation/hardhat-network-helpers");

describe("EvolutionProposal", function () {
    let deployer, evaluator1, evaluator2, evaluator3, proposer, participant1, participant2;
    let evolution, mockRegistry, autonet;

    const CONTENT_CID = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi";
    const REASON_CID = "bafkreiabcdefghijklmnopqrstuvwxyz1234567890abcdef";
    const PROMPT_CID_V1 = "bafyrpb_prompt_v1_placeholder_cid";
    const PROMPT_CID_V2 = "bafyrpb_prompt_v2_updated_cid";

    const MIN_STAKE = ethers.parseEther("100");
    const LARGE_STAKE = ethers.parseEther("500");
    const TRIAL_BUDGET = ethers.parseEther("1000");
    const THREE_DAYS = 3 * 24 * 60 * 60;

    async function deploySetup() {
        // Deploy mock registry
        const MockRegistry = await ethers.getContractFactory("AutonetMockRegistry");
        const mockReg = await MockRegistry.deploy();

        const MockRepToken = await ethers.getContractFactory("AutonetMockRepToken");
        const mockRepToken = await MockRepToken.deploy("Test Rep", "REP");

        // Deploy Autonet as the ATN token
        const Autonet = await ethers.getContractFactory("Autonet");
        const atn = await Autonet.deploy(
            deployer.address,           // economy placeholder
            await mockReg.getAddress(),
            await mockRepToken.getAddress(),
            deployer.address,           // deployer acts as timelock
            "Autonet Credits - Test",
            "ATN-TEST"
        );

        // Deploy EvolutionProposal
        const Evolution = await ethers.getContractFactory("EvolutionProposal");
        const evo = await Evolution.deploy(
            await atn.getAddress(),
            await mockReg.getAddress(),
            deployer.address            // deployer acts as timelock
        );

        return { mockReg, atn, evo };
    }

    async function mintAndApprove(token, user, amount) {
        // Autonet can't directly mint, so we use startEpoch + attestUsage + finalization
        // Simpler: deployer has admin, so we use convertGovernanceToAtn is complex
        // Instead, for testing: register a service, start epoch, attest usage, finalize, claim
        // Actually the simplest: just use the startEpoch and epoch rewards mechanism

        // For test simplicity: give the deployer admin power to directly test
        // We'll have the deployer start an epoch, attest usage, finalize, and transfer

        // Actually let's just use a simple approach: pre-mint via startEpoch
        // Start epoch with a big budget, attest, finalize, the contract holds the tokens
        // The tokens go to the autonet contract itself. We need them in user wallets.

        // The cleanest way: use an ERC20 mock for the ATN token in tests.
        // But EvolutionProposal.sol takes any ERC20 address. Let's deploy a simple test token instead.
    }

    beforeEach(async function () {
        [deployer, evaluator1, evaluator2, evaluator3, proposer, participant1, participant2] =
            await ethers.getSigners();

        // Use a simple TestERC20 for ATN so we can freely mint
        const TestToken = await ethers.getContractFactory("TestERC20");
        const token = await TestToken.deploy("ATN Test", "ATN", 18, 0);
        const tokenAddr = await token.getAddress();

        // Deploy mock registry
        const MockRegistry = await ethers.getContractFactory("AutonetMockRegistry");
        mockRegistry = await MockRegistry.deploy();
        const regAddr = await mockRegistry.getAddress();

        // Deploy EvolutionProposal
        const Evolution = await ethers.getContractFactory("EvolutionProposal");
        evolution = await Evolution.deploy(
            tokenAddr,
            regAddr,
            deployer.address    // deployer acts as timelock
        );

        autonet = token; // Use autonet variable for the token

        // Mint ATN to proposer and deployer
        await autonet.mint(proposer.address, ethers.parseEther("10000"));
        await autonet.mint(deployer.address, ethers.parseEther("10000"));

        // Approve EvolutionProposal to spend tokens
        const evoAddr = await evolution.getAddress();
        await autonet.connect(proposer).approve(evoAddr, ethers.MaxUint256);
        await autonet.connect(deployer).approve(evoAddr, ethers.MaxUint256);

        // Set evaluators
        await evolution.setRPBEvaluator(evaluator1.address, true);
        await evolution.setRPBEvaluator(evaluator2.address, true);
        await evolution.setRPBEvaluator(evaluator3.address, true);
    });

    // =========================================================================
    // PROPOSAL SUBMISSION
    // =========================================================================

    describe("Proposal Submission", function () {
        it("should submit a proposal with stake", async function () {
            await expect(
                evolution.connect(proposer).submitProposal(CONTENT_CID, MIN_STAKE)
            )
                .to.emit(evolution, "ProposalSubmitted")
                .withArgs(1, proposer.address, CONTENT_CID, MIN_STAKE);

            const prop = await evolution.getProposal(1);
            expect(prop.proposer).to.equal(proposer.address);
            expect(prop.contentCid).to.equal(CONTENT_CID);
            expect(prop.stakeAmount).to.equal(MIN_STAKE);
            expect(prop.status).to.equal(0); // Proposed
        });

        it("should transfer stake from proposer to contract", async function () {
            const balBefore = await autonet.balanceOf(proposer.address);
            await evolution.connect(proposer).submitProposal(CONTENT_CID, MIN_STAKE);
            const balAfter = await autonet.balanceOf(proposer.address);

            expect(balBefore - balAfter).to.equal(MIN_STAKE);
            expect(await autonet.balanceOf(await evolution.getAddress())).to.equal(MIN_STAKE);
        });

        it("should reject empty CID", async function () {
            await expect(
                evolution.connect(proposer).submitProposal("", MIN_STAKE)
            ).to.be.revertedWithCustomError(evolution, "InvalidProposal");
        });

        it("should reject insufficient stake", async function () {
            const tooLow = ethers.parseEther("50");
            await expect(
                evolution.connect(proposer).submitProposal(CONTENT_CID, tooLow)
            ).to.be.revertedWithCustomError(evolution, "InsufficientStake");
        });

        it("should increment proposal IDs", async function () {
            await evolution.connect(proposer).submitProposal(CONTENT_CID, MIN_STAKE);
            await evolution.connect(proposer).submitProposal("cid2", MIN_STAKE);

            expect(await evolution.getProposalCount()).to.equal(2);
            const p1 = await evolution.getProposal(1);
            const p2 = await evolution.getProposal(2);
            expect(p1.contentCid).to.equal(CONTENT_CID);
            expect(p2.contentCid).to.equal("cid2");
        });
    });

    // =========================================================================
    // RPB EVALUATION
    // =========================================================================

    describe("RPB Evaluation", function () {
        beforeEach(async function () {
            await evolution.connect(proposer).submitProposal(CONTENT_CID, MIN_STAKE);
            await evolution.beginEvaluation(1);
        });

        it("should move proposal to Evaluating", async function () {
            const prop = await evolution.getProposal(1);
            expect(prop.status).to.equal(1); // Evaluating
        });

        it("should accept evaluator submissions", async function () {
            await expect(
                evolution.connect(evaluator1).submitEvaluation(1, true, 8000, REASON_CID)
            )
                .to.emit(evolution, "RPBEvaluationSubmitted")
                .withArgs(1, evaluator1.address, true, 8000);

            const prop = await evolution.getProposal(1);
            expect(prop.evaluationCount).to.equal(1);
            expect(prop.evaluationScore).to.equal(8000);
        });

        it("should accumulate positive and negative scores", async function () {
            await evolution.connect(evaluator1).submitEvaluation(1, true, 8000, REASON_CID);
            await evolution.connect(evaluator2).submitEvaluation(1, false, 6000, REASON_CID);
            await evolution.connect(evaluator3).submitEvaluation(1, true, 7000, REASON_CID);

            const prop = await evolution.getProposal(1);
            expect(prop.evaluationCount).to.equal(3);
            // +8000 - 6000 + 7000 = 9000
            expect(prop.evaluationScore).to.equal(9000);
        });

        it("should prevent double evaluation", async function () {
            await evolution.connect(evaluator1).submitEvaluation(1, true, 8000, REASON_CID);
            await expect(
                evolution.connect(evaluator1).submitEvaluation(1, false, 5000, REASON_CID)
            ).to.be.revertedWithCustomError(evolution, "AlreadyEvaluated");
        });

        it("should reject non-evaluator submissions", async function () {
            await expect(
                evolution.connect(proposer).submitEvaluation(1, true, 8000, REASON_CID)
            ).to.be.revertedWithCustomError(evolution, "NotAuthorized");
        });

        it("should reject evaluation on non-Evaluating proposal", async function () {
            // Submit new proposal but don't begin evaluation
            await evolution.connect(proposer).submitProposal("cid2", MIN_STAKE);
            await expect(
                evolution.connect(evaluator1).submitEvaluation(2, true, 8000, REASON_CID)
            ).to.be.revertedWithCustomError(evolution, "InvalidStatus");
        });

        it("should reject confidence > 10000", async function () {
            await expect(
                evolution.connect(evaluator1).submitEvaluation(1, true, 10001, REASON_CID)
            ).to.be.revertedWithCustomError(evolution, "InvalidAmount");
        });

        it("should retrieve all evaluations", async function () {
            await evolution.connect(evaluator1).submitEvaluation(1, true, 8000, "cid1");
            await evolution.connect(evaluator2).submitEvaluation(1, false, 6000, "cid2");

            const evals = await evolution.getEvaluations(1);
            expect(evals.evaluatorAddrs.length).to.equal(2);
            expect(evals.approvals[0]).to.equal(true);
            expect(evals.approvals[1]).to.equal(false);
            expect(evals.confidences[0]).to.equal(8000);
            expect(evals.reasonCids[1]).to.equal("cid2");
        });
    });

    // =========================================================================
    // EVALUATION RESOLUTION
    // =========================================================================

    describe("Evaluation Resolution", function () {
        beforeEach(async function () {
            await evolution.connect(proposer).submitProposal(CONTENT_CID, MIN_STAKE);
            await evolution.beginEvaluation(1);
        });

        it("should approve proposal with sufficient approval", async function () {
            // 3 evaluators approve with high confidence
            await evolution.connect(evaluator1).submitEvaluation(1, true, 8000, REASON_CID);
            await evolution.connect(evaluator2).submitEvaluation(1, true, 7000, REASON_CID);
            await evolution.connect(evaluator3).submitEvaluation(1, true, 9000, REASON_CID);

            // Advance time past evaluation period
            await time.increase(THREE_DAYS + 1);

            await expect(evolution.resolveEvaluation(1))
                .to.emit(evolution, "ProposalStatusChanged")
                .withArgs(1, 1, 2); // Evaluating -> Trial

            const prop = await evolution.getProposal(1);
            expect(prop.status).to.equal(2); // Trial
        });

        it("should reject proposal with insufficient approval", async function () {
            // 2 reject, 1 approve
            await evolution.connect(evaluator1).submitEvaluation(1, false, 8000, REASON_CID);
            await evolution.connect(evaluator2).submitEvaluation(1, false, 7000, REASON_CID);
            await evolution.connect(evaluator3).submitEvaluation(1, true, 5000, REASON_CID);

            await time.increase(THREE_DAYS + 1);

            await evolution.resolveEvaluation(1);
            const prop = await evolution.getProposal(1);
            expect(prop.status).to.equal(4); // Rejected
        });

        it("should revert before evaluation period ends", async function () {
            await evolution.connect(evaluator1).submitEvaluation(1, true, 8000, REASON_CID);
            await evolution.connect(evaluator2).submitEvaluation(1, true, 7000, REASON_CID);
            await evolution.connect(evaluator3).submitEvaluation(1, true, 9000, REASON_CID);

            // Don't advance time
            await expect(evolution.resolveEvaluation(1))
                .to.be.revertedWithCustomError(evolution, "EvaluationPeriodActive");
        });

        it("should revert without quorum", async function () {
            // Only 1 evaluation (quorum is 3)
            await evolution.connect(evaluator1).submitEvaluation(1, true, 8000, REASON_CID);

            await time.increase(THREE_DAYS + 1);

            await expect(evolution.resolveEvaluation(1))
                .to.be.revertedWithCustomError(evolution, "QuorumNotReached");
        });

        it("should refund stake minus fee on rejection", async function () {
            await evolution.connect(evaluator1).submitEvaluation(1, false, 8000, REASON_CID);
            await evolution.connect(evaluator2).submitEvaluation(1, false, 7000, REASON_CID);
            await evolution.connect(evaluator3).submitEvaluation(1, false, 9000, REASON_CID);

            await time.increase(THREE_DAYS + 1);

            const balBefore = await autonet.balanceOf(proposer.address);
            await evolution.resolveEvaluation(1);
            const balAfter = await autonet.balanceOf(proposer.address);

            // 5% fee: refund = 100 * 0.95 = 95 ATN
            const fee = (MIN_STAKE * 500n) / 10000n;
            const expectedRefund = MIN_STAKE - fee;
            expect(balAfter - balBefore).to.equal(expectedRefund);
        });
    });

    // =========================================================================
    // TRIAL MANAGEMENT
    // =========================================================================

    describe("Trial Management", function () {
        beforeEach(async function () {
            await evolution.connect(proposer).submitProposal(CONTENT_CID, MIN_STAKE);
            await evolution.beginEvaluation(1);

            // All approve
            await evolution.connect(evaluator1).submitEvaluation(1, true, 8000, REASON_CID);
            await evolution.connect(evaluator2).submitEvaluation(1, true, 7000, REASON_CID);
            await evolution.connect(evaluator3).submitEvaluation(1, true, 9000, REASON_CID);

            await time.increase(THREE_DAYS + 1);
            await evolution.resolveEvaluation(1);
        });

        it("should fund a trial", async function () {
            await expect(evolution.fundTrial(1, TRIAL_BUDGET))
                .to.emit(evolution, "TrialFunded")
                .withArgs(1, TRIAL_BUDGET);

            const prop = await evolution.getProposal(1);
            expect(prop.trialBudget).to.equal(TRIAL_BUDGET);
        });

        it("should record trial participants", async function () {
            await evolution.fundTrial(1, TRIAL_BUDGET);

            await expect(
                evolution.recordTrialContribution(1, participant1.address, 0, 100) // Compute
            )
                .to.emit(evolution, "TrialParticipantAdded")
                .withArgs(1, participant1.address, 0);

            await evolution.recordTrialContribution(1, participant2.address, 2, 50); // Proposal

            const parts = await evolution.getTrialParticipants(1);
            expect(parts.participants.length).to.equal(2);
            expect(parts.participants[0]).to.equal(participant1.address);
            expect(parts.types[1]).to.equal(2); // Proposal
        });

        it("should reject funding non-Trial proposal", async function () {
            await evolution.connect(proposer).submitProposal("cid2", MIN_STAKE);
            await expect(evolution.fundTrial(2, TRIAL_BUDGET))
                .to.be.revertedWithCustomError(evolution, "InvalidStatus");
        });
    });

    // =========================================================================
    // ADOPTION & REWARD CLAIMING
    // =========================================================================

    describe("Adoption & Reward Claiming", function () {
        beforeEach(async function () {
            await evolution.connect(proposer).submitProposal(CONTENT_CID, MIN_STAKE);
            await evolution.beginEvaluation(1);

            await evolution.connect(evaluator1).submitEvaluation(1, true, 8000, REASON_CID);
            await evolution.connect(evaluator2).submitEvaluation(1, true, 7000, REASON_CID);
            await evolution.connect(evaluator3).submitEvaluation(1, true, 9000, REASON_CID);

            await time.increase(THREE_DAYS + 1);
            await evolution.resolveEvaluation(1);

            await evolution.fundTrial(1, TRIAL_BUDGET);

            // Record contributions with different types
            await evolution.recordTrialContribution(1, participant1.address, 0, 100); // Compute, 100 units
            await evolution.recordTrialContribution(1, participant2.address, 2, 50);  // Proposal, 50 units
        });

        it("should adopt and return proposer stake", async function () {
            const balBefore = await autonet.balanceOf(proposer.address);

            await expect(evolution.adoptProposal(1))
                .to.emit(evolution, "ProposalAdopted");

            const balAfter = await autonet.balanceOf(proposer.address);
            expect(balAfter - balBefore).to.equal(MIN_STAKE); // Stake returned

            const prop = await evolution.getProposal(1);
            expect(prop.status).to.equal(3); // Adopted
        });

        it("should distribute rewards weighted by contribution type", async function () {
            await evolution.adoptProposal(1);

            // Compute: weight 10000, units 100 => weighted = 1,000,000
            // Proposal: weight 30000, units 50 => weighted = 1,500,000
            // Total weighted = 2,500,000
            // participant1 gets 1,000,000 / 2,500,000 * 1000 = 400 ATN
            // participant2 gets 1,500,000 / 2,500,000 * 1000 = 600 ATN

            const bal1Before = await autonet.balanceOf(participant1.address);
            await evolution.connect(participant1).claimTrialReward(1, 0);
            const bal1After = await autonet.balanceOf(participant1.address);

            const bal2Before = await autonet.balanceOf(participant2.address);
            await evolution.connect(participant2).claimTrialReward(1, 1);
            const bal2After = await autonet.balanceOf(participant2.address);

            // participant1: Compute 100 * 10000 = 1,000,000 weighted
            // participant2: Proposal 50 * 30000 = 1,500,000 weighted
            // total = 2,500,000
            // participant1 reward = 1,000,000 / 2,500,000 * 1000e18 = 400e18
            // participant2 reward = 1,500,000 / 2,500,000 * 1000e18 = 600e18
            expect(bal1After - bal1Before).to.equal(ethers.parseEther("400"));
            expect(bal2After - bal2Before).to.equal(ethers.parseEther("600"));
        });

        it("should prevent double claiming", async function () {
            await evolution.adoptProposal(1);
            await evolution.connect(participant1).claimTrialReward(1, 0);

            await expect(
                evolution.connect(participant1).claimTrialReward(1, 0)
            ).to.be.revertedWithCustomError(evolution, "AlreadyClaimed");
        });

        it("should reject claiming from non-participant", async function () {
            await evolution.adoptProposal(1);
            await expect(
                evolution.connect(deployer).claimTrialReward(1, 0) // deployer is not participant1
            ).to.be.revertedWithCustomError(evolution, "NotAuthorized");
        });

        it("should reject claiming from non-adopted proposal", async function () {
            // Proposal is in Trial, not Adopted
            await expect(
                evolution.connect(participant1).claimTrialReward(1, 0)
            ).to.be.revertedWithCustomError(evolution, "InvalidStatus");
        });
    });

    // =========================================================================
    // REJECTION
    // =========================================================================

    describe("Rejection", function () {
        it("should allow admin to reject at any non-terminal status", async function () {
            await evolution.connect(proposer).submitProposal(CONTENT_CID, MIN_STAKE);

            const balBefore = await autonet.balanceOf(proposer.address);
            await expect(evolution.rejectProposal(1))
                .to.emit(evolution, "ProposalRejected");
            const balAfter = await autonet.balanceOf(proposer.address);

            const fee = (MIN_STAKE * 500n) / 10000n;
            expect(balAfter - balBefore).to.equal(MIN_STAKE - fee);
        });

        it("should not reject already adopted proposal", async function () {
            await evolution.connect(proposer).submitProposal(CONTENT_CID, MIN_STAKE);
            await evolution.beginEvaluation(1);
            await evolution.connect(evaluator1).submitEvaluation(1, true, 8000, REASON_CID);
            await evolution.connect(evaluator2).submitEvaluation(1, true, 7000, REASON_CID);
            await evolution.connect(evaluator3).submitEvaluation(1, true, 9000, REASON_CID);
            await time.increase(THREE_DAYS + 1);
            await evolution.resolveEvaluation(1);
            await evolution.fundTrial(1, TRIAL_BUDGET);
            await evolution.adoptProposal(1);

            await expect(evolution.rejectProposal(1))
                .to.be.revertedWithCustomError(evolution, "InvalidStatus");
        });

        it("should not reject already rejected proposal", async function () {
            await evolution.connect(proposer).submitProposal(CONTENT_CID, MIN_STAKE);
            await evolution.rejectProposal(1);

            await expect(evolution.rejectProposal(1))
                .to.be.revertedWithCustomError(evolution, "InvalidStatus");
        });
    });

    // =========================================================================
    // RPB PROMPT MANAGEMENT
    // =========================================================================

    describe("RPB Prompt Management", function () {
        it("should update RPB prompt via timelock", async function () {
            await expect(evolution.updateRPBPrompt(PROMPT_CID_V1))
                .to.emit(evolution, "RPBPromptUpdated")
                .withArgs(1, PROMPT_CID_V1);

            expect(await evolution.currentPromptVersion()).to.equal(1);
            expect(await evolution.getCurrentPromptCid()).to.equal(PROMPT_CID_V1);
        });

        it("should version prompts", async function () {
            await evolution.updateRPBPrompt(PROMPT_CID_V1);
            await evolution.updateRPBPrompt(PROMPT_CID_V2);

            expect(await evolution.currentPromptVersion()).to.equal(2);
            expect(await evolution.getCurrentPromptCid()).to.equal(PROMPT_CID_V2);
            expect(await evolution.getPromptVersion(1)).to.equal(PROMPT_CID_V1);
            expect(await evolution.getPromptVersion(2)).to.equal(PROMPT_CID_V2);
        });

        it("should store prompts in Registry", async function () {
            await evolution.updateRPBPrompt(PROMPT_CID_V1);

            // Verify Registry was updated
            expect(await mockRegistry.getRegistryValue("rpb.prompt.current")).to.equal(PROMPT_CID_V1);
            expect(await mockRegistry.getRegistryValue("rpb.prompt.v1")).to.equal(PROMPT_CID_V1);
            expect(await mockRegistry.getRegistryValue("rpb.prompt.version")).to.equal("1");
        });

        it("should reject empty prompt CID", async function () {
            await expect(evolution.updateRPBPrompt(""))
                .to.be.revertedWithCustomError(evolution, "InvalidProposal");
        });

        it("should only allow timelock to update prompt", async function () {
            await expect(
                evolution.connect(proposer).updateRPBPrompt(PROMPT_CID_V1)
            ).to.be.revertedWithCustomError(evolution, "NotAuthorized");
        });
    });

    // =========================================================================
    // CONTRIBUTION WEIGHTS
    // =========================================================================

    describe("Contribution Weights", function () {
        it("should have correct default weights", async function () {
            const weights = await evolution.getContributionWeights();
            expect(weights.compute).to.equal(10000);    // 1x
            expect(weights.diagnosis).to.equal(15000);   // 1.5x
            expect(weights.proposal_).to.equal(30000);   // 3x
            expect(weights.validation).to.equal(12000);  // 1.2x
        });

        it("should allow admin to update weights", async function () {
            await expect(
                evolution.setContributionWeights(10000, 20000, 40000, 15000)
            ).to.emit(evolution, "ContributionWeightsUpdated");

            const weights = await evolution.getContributionWeights();
            expect(weights.diagnosis).to.equal(20000);
            expect(weights.proposal_).to.equal(40000);
        });

        it("should reject zero compute weight", async function () {
            await expect(
                evolution.setContributionWeights(0, 20000, 40000, 15000)
            ).to.be.revertedWithCustomError(evolution, "InvalidAmount");
        });
    });

    // =========================================================================
    // CONFIGURATION
    // =========================================================================

    describe("Configuration", function () {
        it("should allow admin to set min stake", async function () {
            await evolution.setMinStake(ethers.parseEther("200"));
            expect(await evolution.minStake()).to.equal(ethers.parseEther("200"));
        });

        it("should allow admin to set processing fee", async function () {
            await evolution.setProcessingFeeBps(1000);
            expect(await evolution.processingFeeBps()).to.equal(1000);
        });

        it("should reject processing fee > 50%", async function () {
            await expect(evolution.setProcessingFeeBps(5001))
                .to.be.revertedWithCustomError(evolution, "InvalidAmount");
        });

        it("should allow admin to set evaluation quorum", async function () {
            await evolution.setEvaluationQuorum(5);
            expect(await evolution.evaluationQuorum()).to.equal(5);
        });

        it("should allow admin to set approval threshold", async function () {
            await evolution.setApprovalThreshold(7500);
            expect(await evolution.approvalThresholdBps()).to.equal(7500);
        });

        it("should allow admin to set evaluation period", async function () {
            await evolution.setEvaluationPeriod(7 * 24 * 60 * 60); // 7 days
            expect(await evolution.evaluationPeriod()).to.equal(7 * 24 * 60 * 60);
        });

        it("should allow admin to manage evaluators", async function () {
            const newEval = participant1.address;
            await expect(evolution.setRPBEvaluator(newEval, true))
                .to.emit(evolution, "RPBEvaluatorSet")
                .withArgs(newEval, true);

            expect(await evolution.rpbEvaluators(newEval)).to.equal(true);
        });

        it("should allow admin to transfer admin", async function () {
            await evolution.transferAdmin(proposer.address);
            expect(await evolution.admin()).to.equal(proposer.address);
        });

        it("should allow admin to recover tokens", async function () {
            // Some tokens are in the contract from a rejected proposal
            await evolution.connect(proposer).submitProposal(CONTENT_CID, MIN_STAKE);
            await evolution.rejectProposal(1);

            // Fee stays in contract
            const fee = (MIN_STAKE * 500n) / 10000n;
            const contractBal = await autonet.balanceOf(await evolution.getAddress());
            expect(contractBal).to.equal(fee);

            await evolution.recoverTokens(deployer.address, fee);
            expect(await autonet.balanceOf(await evolution.getAddress())).to.equal(0);
        });
    });

    // =========================================================================
    // DAO INTEGRATION
    // =========================================================================

    describe("DAO Integration", function () {
        it("should link a DAO proposal to an evolution proposal", async function () {
            await evolution.connect(proposer).submitProposal(CONTENT_CID, MIN_STAKE);
            await evolution.linkDAOProposal(1, 42);

            const prop = await evolution.getProposal(1);
            expect(prop.daoProposalId).to.equal(42);
        });
    });

    // =========================================================================
    // FULL LIFECYCLE
    // =========================================================================

    describe("Full Lifecycle", function () {
        it("should complete propose -> evaluate -> trial -> adopt -> claim", async function () {
            // 1. Submit proposal
            await evolution.connect(proposer).submitProposal(CONTENT_CID, LARGE_STAKE);

            // 2. Begin evaluation
            await evolution.beginEvaluation(1);

            // 3. RPB evaluates
            await evolution.connect(evaluator1).submitEvaluation(1, true, 9000, "r1");
            await evolution.connect(evaluator2).submitEvaluation(1, true, 8500, "r2");
            await evolution.connect(evaluator3).submitEvaluation(1, true, 7500, "r3");

            // 4. Resolve (after period)
            await time.increase(THREE_DAYS + 1);
            await evolution.resolveEvaluation(1);

            // 5. Fund trial
            await evolution.fundTrial(1, TRIAL_BUDGET);

            // 6. Record contributions
            // Compute: 200 units * 10000 weight = 2,000,000
            // Diagnosis: 100 units * 15000 weight = 1,500,000
            // Proposal: 50 units * 30000 weight = 1,500,000
            // Total weighted: 5,000,000
            await evolution.recordTrialContribution(1, participant1.address, 0, 200); // Compute
            await evolution.recordTrialContribution(1, participant1.address, 1, 100); // Diagnosis
            await evolution.recordTrialContribution(1, participant2.address, 2, 50);  // Proposal

            // 7. Adopt
            await evolution.adoptProposal(1);

            // Verify proposer got stake back
            const prop = await evolution.getProposal(1);
            expect(prop.status).to.equal(3); // Adopted

            // 8. Claim rewards
            // participant1 (Compute): 2,000,000 / 5,000,000 * 1000 = 400 ATN
            const bal1Before = await autonet.balanceOf(participant1.address);
            await evolution.connect(participant1).claimTrialReward(1, 0);
            const bal1After = await autonet.balanceOf(participant1.address);
            expect(bal1After - bal1Before).to.equal(ethers.parseEther("400"));

            // participant1 (Diagnosis): 1,500,000 / 5,000,000 * 1000 = 300 ATN
            await evolution.connect(participant1).claimTrialReward(1, 1);
            const bal1Final = await autonet.balanceOf(participant1.address);
            expect(bal1Final - bal1After).to.equal(ethers.parseEther("300"));

            // participant2 (Proposal): 1,500,000 / 5,000,000 * 1000 = 300 ATN
            const bal2Before = await autonet.balanceOf(participant2.address);
            await evolution.connect(participant2).claimTrialReward(1, 2);
            const bal2After = await autonet.balanceOf(participant2.address);
            expect(bal2After - bal2Before).to.equal(ethers.parseEther("300"));
        });
    });
});
