const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("CapabilityScorecard", function () {
    let deployer, evaluator1, evaluator2, user1;
    let scorecard;

    const VISUAL_ENCODER = ethers.keccak256(ethers.toUtf8Bytes("visual_encoder"));
    const TEXT_DECODER = ethers.keccak256(ethers.toUtf8Bytes("text_decoder"));
    const CROSS_MODAL = ethers.keccak256(ethers.toUtf8Bytes("cross_modal_fusion"));

    beforeEach(async function () {
        [deployer, evaluator1, evaluator2, user1] = await ethers.getSigners();

        const Scorecard = await ethers.getContractFactory("CapabilityScorecard");
        scorecard = await Scorecard.deploy(deployer.address);

        await scorecard.setEvaluator(evaluator1.address, true);
    });

    describe("Module Registration", function () {
        it("should register a module with target score", async function () {
            await expect(scorecard.registerModule(VISUAL_ENCODER, 8000))
                .to.emit(scorecard, "ModuleRegistered")
                .withArgs(VISUAL_ENCODER, 8000);

            const mod = await scorecard.modules(VISUAL_ENCODER);
            expect(mod.targetScore).to.equal(8000);
            expect(mod.score).to.equal(0);
        });

        it("should not allow duplicate registration", async function () {
            await scorecard.registerModule(VISUAL_ENCODER, 8000);
            await expect(scorecard.registerModule(VISUAL_ENCODER, 8000))
                .to.be.revertedWithCustomError(scorecard, "ModuleAlreadyExists");
        });

        it("should reject invalid target scores", async function () {
            await expect(scorecard.registerModule(VISUAL_ENCODER, 0))
                .to.be.revertedWithCustomError(scorecard, "InvalidScore");
            await expect(scorecard.registerModule(VISUAL_ENCODER, 10001))
                .to.be.revertedWithCustomError(scorecard, "InvalidScore");
        });

        it("should only allow admin to register", async function () {
            await expect(
                scorecard.connect(user1).registerModule(VISUAL_ENCODER, 8000)
            ).to.be.revertedWithCustomError(scorecard, "NotAuthorized");
        });
    });

    describe("Score Updates", function () {
        beforeEach(async function () {
            await scorecard.registerModule(VISUAL_ENCODER, 8000);
        });

        it("should update score on first evaluation", async function () {
            await expect(scorecard.connect(evaluator1).updateScore(VISUAL_ENCODER, 3000))
                .to.emit(scorecard, "ScoreUpdated")
                .withArgs(VISUAL_ENCODER, 0, 3000, evaluator1.address);

            const mod = await scorecard.modules(VISUAL_ENCODER);
            expect(mod.score).to.equal(3000);
            expect(mod.evaluationCount).to.equal(1);
        });

        it("should use EMA for subsequent evaluations", async function () {
            // First: score = 4000
            await scorecard.connect(evaluator1).updateScore(VISUAL_ENCODER, 4000);
            // Second: EMA = (4000 * 1 + 6000) / 2 = 5000
            await scorecard.connect(evaluator1).updateScore(VISUAL_ENCODER, 6000);

            const mod = await scorecard.modules(VISUAL_ENCODER);
            expect(mod.score).to.equal(5000);
            expect(mod.evaluationCount).to.equal(2);
        });

        it("should reject updates from non-evaluators", async function () {
            await expect(
                scorecard.connect(user1).updateScore(VISUAL_ENCODER, 5000)
            ).to.be.revertedWithCustomError(scorecard, "NotAuthorized");
        });

        it("should reject score above max", async function () {
            await expect(
                scorecard.connect(evaluator1).updateScore(VISUAL_ENCODER, 10001)
            ).to.be.revertedWithCustomError(scorecard, "InvalidScore");
        });

        it("should reject update for unregistered module", async function () {
            await expect(
                scorecard.connect(evaluator1).updateScore(TEXT_DECODER, 5000)
            ).to.be.revertedWithCustomError(scorecard, "ModuleNotFound");
        });
    });

    describe("Reward Multipliers", function () {
        beforeEach(async function () {
            await scorecard.registerModule(VISUAL_ENCODER, 8000);
        });

        it("should return maxMultiplier (3x) for zero-score module", async function () {
            // Score is 0, target is 8000 → max multiplier
            const multiplier = await scorecard.getRewardMultiplier(VISUAL_ENCODER);
            expect(multiplier).to.equal(30000); // 3x
        });

        it("should return minMultiplier (0.5x) for saturated module", async function () {
            // Set score to target
            await scorecard.connect(evaluator1).updateScore(VISUAL_ENCODER, 8000);
            const multiplier = await scorecard.getRewardMultiplier(VISUAL_ENCODER);
            expect(multiplier).to.equal(5000); // 0.5x
        });

        it("should return minMultiplier for above-target score", async function () {
            await scorecard.connect(evaluator1).updateScore(VISUAL_ENCODER, 10000);
            const multiplier = await scorecard.getRewardMultiplier(VISUAL_ENCODER);
            expect(multiplier).to.equal(5000);
        });

        it("should interpolate for mid-range score", async function () {
            // Score = 4000, Target = 8000 → 50% of the way
            // Multiplier = 30000 - (50% * (30000 - 5000)) = 30000 - 12500 = 17500
            await scorecard.connect(evaluator1).updateScore(VISUAL_ENCODER, 4000);
            const multiplier = await scorecard.getRewardMultiplier(VISUAL_ENCODER);
            expect(multiplier).to.equal(17500); // 1.75x
        });

        it("should return base multiplier for unknown module", async function () {
            const unknown = ethers.keccak256(ethers.toUtf8Bytes("unknown"));
            const multiplier = await scorecard.getRewardMultiplier(unknown);
            expect(multiplier).to.equal(10000); // 1x base
        });
    });

    describe("Scorecard View", function () {
        it("should return full scorecard", async function () {
            await scorecard.registerModule(VISUAL_ENCODER, 8000);
            await scorecard.registerModule(TEXT_DECODER, 6000);

            await scorecard.connect(evaluator1).updateScore(VISUAL_ENCODER, 4000);
            // TEXT_DECODER stays at 0

            const result = await scorecard.getScorecard();
            expect(result.ids.length).to.equal(2);
            expect(result.scores[0]).to.equal(4000);
            expect(result.scores[1]).to.equal(0);
            expect(result.targets[0]).to.equal(8000);
            expect(result.targets[1]).to.equal(6000);
            expect(result.multipliers[0]).to.equal(17500); // 1.75x
            expect(result.multipliers[1]).to.equal(30000); // 3x (no capability)
        });
    });

    describe("Evaluator Management", function () {
        it("should add and remove evaluators", async function () {
            await scorecard.setEvaluator(evaluator2.address, true);
            expect(await scorecard.evaluators(evaluator2.address)).to.equal(true);

            await scorecard.setEvaluator(evaluator2.address, false);
            expect(await scorecard.evaluators(evaluator2.address)).to.equal(false);
        });
    });
});

describe("Dynamic Training Pricing (Autonet + Scorecard)", function () {
    let deployer, user1, user2;
    let autonet, scorecard;

    const VISUAL_ENCODER = ethers.keccak256(ethers.toUtf8Bytes("visual_encoder"));
    const TEXT_DECODER = ethers.keccak256(ethers.toUtf8Bytes("text_decoder"));

    async function deployMockJurisdiction() {
        const MockRegistry = await ethers.getContractFactory("AutonetMockRegistry");
        const mockRegistry = await MockRegistry.deploy();
        const MockRepToken = await ethers.getContractFactory("AutonetMockRepToken");
        const mockRepToken = await MockRepToken.deploy("Test Rep", "REP");

        return {
            economy: deployer.address,
            registry: await mockRegistry.getAddress(),
            repToken: await mockRepToken.getAddress(),
            timelock: deployer.address,
        };
    }

    beforeEach(async function () {
        [deployer, user1, user2] = await ethers.getSigners();

        const jurisdiction = await deployMockJurisdiction();

        const Autonet = await ethers.getContractFactory("Autonet");
        autonet = await Autonet.deploy(
            jurisdiction.economy,
            jurisdiction.registry,
            jurisdiction.repToken,
            jurisdiction.timelock,
            "ATN Test",
            "ATN"
        );

        const Scorecard = await ethers.getContractFactory("CapabilityScorecard");
        scorecard = await Scorecard.deploy(deployer.address);

        // Wire scorecard into Autonet
        await autonet.setCapabilityScorecard(await scorecard.getAddress());

        // Register modules
        await scorecard.registerModule(VISUAL_ENCODER, 8000);
        await scorecard.registerModule(TEXT_DECODER, 6000);

        // Set visual_encoder to 50% capability, text_decoder stays at 0%
        await scorecard.updateScore(VISUAL_ENCODER, 4000);
    });

    it("should weight rewards by capability multiplier", async function () {
        const svcVisual = ethers.keccak256(ethers.toUtf8Bytes("svc-visual"));
        const svcText = ethers.keccak256(ethers.toUtf8Bytes("svc-text"));
        const budget = ethers.parseEther("1000");

        // Register and activate both services
        await autonet.connect(user1).registerService(svcVisual, user1.address, "h1");
        await autonet.connect(user2).registerService(svcText, user2.address, "h2");
        await autonet.activateService(svcVisual);
        await autonet.activateService(svcText);

        // Associate services with modules
        await autonet.setServiceModule(svcVisual, VISUAL_ENCODER);
        await autonet.setServiceModule(svcText, TEXT_DECODER);

        // Start epoch and attest equal usage
        await autonet.startEpoch(budget);
        await autonet.connect(user1).attestUsage(svcVisual, 100);
        await autonet.connect(user2).attestUsage(svcText, 100);

        await autonet.finalizeCurrentEpoch();

        // Visual encoder: score=4000, target=8000 → multiplier=17500 (1.75x)
        // Text decoder:   score=0,    target=6000 → multiplier=30000 (3x)
        // Weighted visual: 100 * 17500 / 10000 = 175
        // Weighted text:   100 * 30000 / 10000 = 300
        // Total weighted: 475
        // Visual reward: 175/475 * 1000 ≈ 368.42 ATN
        // Text reward:   300/475 * 1000 ≈ 631.57 ATN

        const rewardVisual = await autonet.getServiceEpochReward(svcVisual, 1);
        const rewardText = await autonet.getServiceEpochReward(svcText, 1);

        // Text (0% capability, 3x multiplier) should get more than visual (50% capability, 1.75x)
        expect(rewardText.totalReward).to.be.gt(rewardVisual.totalReward);

        // Approximate ratios (within rounding)
        const total = rewardVisual.totalReward + rewardText.totalReward;
        // total should approximately equal budget (rounding may lose small fractions)
        expect(total).to.be.closeTo(budget, ethers.parseEther("1"));
    });

    it("should distribute equally without scorecard", async function () {
        // Remove scorecard
        await autonet.setCapabilityScorecard(ethers.ZeroAddress);

        const svcA = ethers.keccak256(ethers.toUtf8Bytes("svc-a"));
        const svcB = ethers.keccak256(ethers.toUtf8Bytes("svc-b"));
        const budget = ethers.parseEther("1000");

        await autonet.connect(user1).registerService(svcA, user1.address, "h1");
        await autonet.connect(user2).registerService(svcB, user2.address, "h2");
        await autonet.activateService(svcA);
        await autonet.activateService(svcB);

        await autonet.startEpoch(budget);
        await autonet.connect(user1).attestUsage(svcA, 100);
        await autonet.connect(user2).attestUsage(svcB, 100);

        await autonet.finalizeCurrentEpoch();

        const rewardA = await autonet.getServiceEpochReward(svcA, 1);
        const rewardB = await autonet.getServiceEpochReward(svcB, 1);

        // Equal usage, no scorecard → equal rewards
        expect(rewardA.totalReward).to.equal(ethers.parseEther("500"));
        expect(rewardB.totalReward).to.equal(ethers.parseEther("500"));
    });

    it("should handle service without module assignment (1x multiplier)", async function () {
        const svcUnassigned = ethers.keccak256(ethers.toUtf8Bytes("svc-no-module"));
        const svcText = ethers.keccak256(ethers.toUtf8Bytes("svc-text"));
        const budget = ethers.parseEther("1000");

        await autonet.connect(user1).registerService(svcUnassigned, user1.address, "h1");
        await autonet.connect(user2).registerService(svcText, user2.address, "h2");
        await autonet.activateService(svcUnassigned);
        await autonet.activateService(svcText);

        // Only assign svcText to a module
        await autonet.setServiceModule(svcText, TEXT_DECODER);

        await autonet.startEpoch(budget);
        await autonet.connect(user1).attestUsage(svcUnassigned, 100);
        await autonet.connect(user2).attestUsage(svcText, 100);

        await autonet.finalizeCurrentEpoch();

        const rewardUnassigned = await autonet.getServiceEpochReward(svcUnassigned, 1);
        const rewardText = await autonet.getServiceEpochReward(svcText, 1);

        // Unassigned: 100 * 1x = 100 weighted
        // Text: 100 * 3x = 300 weighted
        // Unassigned: 100/400 * 1000 = 250
        // Text: 300/400 * 1000 = 750
        expect(rewardUnassigned.totalReward).to.equal(ethers.parseEther("250"));
        expect(rewardText.totalReward).to.equal(ethers.parseEther("750"));
    });
});
